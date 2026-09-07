r"""CART decision tree — classification.

Greedy recursive binary partitioning (Breiman et al., 1984). Every internal
node tests ``x[feature] <= threshold``; the split chosen at a node is the
one that maximises the impurity decrease

.. math::
    \Delta H(j, \tau) = H(S_t)
      - \frac{n_L}{n_t} H(S_L) - \frac{n_R}{n_t} H(S_R)

over all features ``j`` and midpoint thresholds ``tau``, where ``H`` is the
Gini index :math:`1 - \sum_k p_k^2` or the Shannon entropy in bits
:math:`-\sum_k p_k \log_2 p_k`. Recursion stops at ``max_depth``, at
``min_samples_split``, at a pure node, or when no split clears
``min_impurity_decrease`` (scaled as :math:`\frac{n_t}{n}\Delta H`, to match
scikit-learn). Leaves predict their class-frequency vector.

Building the optimal tree is NP-complete; this greedy heuristic is the
standard practical substitute. There is no cost-complexity pruning
(``ccp_alpha``) and no feature subsampling (``max_features``) — see
``docs/derivations/decision_tree.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.typing import FeatureMatrix, FloatArray, IntArray, TargetVector
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y

# A split must beat the incumbent by more than this to be taken, which also
# rejects numerically-spurious "gains" of order machine epsilon on a node
# that carries no real information.
_MIN_GAIN = 1e-12


def _gini(counts: FloatArray) -> FloatArray:
    r"""Gini impurity :math:`1 - \sum_k p_k^2` from class counts.

    ``counts`` is ``(..., n_classes)``; the impurity is computed over the
    last axis, so a ``(n_classes,)`` vector returns a scalar and a
    ``(m, n_classes)`` matrix returns ``(m,)`` — one impurity per row.
    """
    total = counts.sum(axis=-1, keepdims=True)
    total = np.where(total == 0.0, 1.0, total)  # empty partition -> impurity 0
    p = counts / total  # p_k
    return 1.0 - np.sum(p**2, axis=-1)


def _entropy(counts: FloatArray) -> FloatArray:
    r"""Shannon entropy in bits :math:`-\sum_k p_k \log_2 p_k` from class counts.

    Same broadcasting rule as :func:`_gini`. ``0 log 0`` is taken as ``0``.
    """
    total = counts.sum(axis=-1, keepdims=True)
    total = np.where(total == 0.0, 1.0, total)
    p = counts / total  # p_k
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0.0, p * np.log2(p), 0.0)  # p_k log2 p_k, with 0 log 0 = 0
    return -np.sum(terms, axis=-1)


_Criterion = Callable[[FloatArray], FloatArray]
_CRITERIA: dict[str, _Criterion] = {"gini": _gini, "entropy": _entropy}


def _best_split(
    X_node: FloatArray,
    y_idx: IntArray,
    n_classes: int,
    impurity_fn: _Criterion,
    min_samples_leaf: int,
) -> tuple[int | None, float, float]:
    r"""Best ``(feature, threshold)`` split of a node, and its impurity decrease.

    Sweeps every feature: sort the node's rows by that feature, let the
    left partition grow one row at a time (its class counts are the running
    cumulative sum of the one-hot labels), and score the split at each
    midpoint between distinct adjacent values. Returns ``(None, 0.0, 0.0)``
    if no valid split exists (every feature constant, or none leaves both
    children with ``min_samples_leaf`` rows).

    Ties are broken towards the lowest feature index, then the lowest
    threshold: a later split must beat the incumbent gain by more than
    ``_MIN_GAIN`` to replace it.
    """
    n_t = X_node.shape[0]
    parent_counts = np.bincount(y_idx, minlength=n_classes).astype(np.float64)
    parent_impurity = float(impurity_fn(parent_counts))  # H(S_t)
    one_hot = np.eye(n_classes)[y_idx]  # (n_t, n_classes)

    n_left = np.arange(1, n_t)  # left-partition size at each candidate m
    n_right = n_t - n_left

    best_gain, best_feature, best_threshold = 0.0, None, 0.0
    for j in range(X_node.shape[1]):
        order = np.argsort(X_node[:, j], kind="stable")
        x_sorted = X_node[order, j]
        left_counts = np.cumsum(one_hot[order], axis=0)[:-1]  # rows order[:m+1]
        right_counts = parent_counts - left_counts  # (n_t - 1, n_classes)

        # weighted child impurity at every m:  (n_L/n_t) H(S_L) + (n_R/n_t) H(S_R)
        child_impurity = (
            n_left * impurity_fn(left_counts) + n_right * impurity_fn(right_counts)
        ) / n_t
        gains = parent_impurity - child_impurity  # ΔH at every m

        # a split is valid only strictly between two distinct values and only
        # if both children keep at least min_samples_leaf rows
        valid = (
            (x_sorted[:-1] != x_sorted[1:])
            & (n_left >= min_samples_leaf)
            & (n_right >= min_samples_leaf)
        )
        if not valid.any():
            continue
        gains = np.where(valid, gains, -np.inf)
        m = int(np.argmax(gains))  # first maximal m -> lowest threshold
        if gains[m] > best_gain + _MIN_GAIN:
            best_gain = float(gains[m])
            best_feature = j
            best_threshold = 0.5 * float(x_sorted[m] + x_sorted[m + 1])  # midpoint

    return best_feature, best_threshold, best_gain


@dataclass
class _Node:
    """One node of a fitted tree — internal split or leaf.

    A node is a leaf iff ``left`` is ``None``. ``value`` is the
    class-frequency vector of the training samples that reached it (columns
    ordered as ``classes_``); ``impurity`` and ``n_samples`` are recorded
    for inspection and for ``feature_importances_``.
    """

    n_samples: int
    impurity: float
    value: FloatArray
    feature: int | None = None
    threshold: float | None = None
    left: _Node | None = None
    right: _Node | None = None

    @property
    def is_leaf(self) -> bool:
        """``True`` when this node has no children."""
        return self.left is None


class DecisionTreeClassifier(Estimator):
    r"""A CART decision tree for classification.

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        Impurity measure a split is scored on. ``"gini"`` is
        :math:`1 - \sum_k p_k^2`; ``"entropy"`` is the Shannon entropy in
        bits, :math:`-\sum_k p_k \log_2 p_k` (scikit-learn's convention).
    max_depth : int, optional
        Maximum depth of the tree. ``None`` (the default) grows nodes until
        they are pure or too small to split. The root alone has depth 0.
    min_samples_split : int, default=2
        A node with fewer samples than this becomes a leaf.
    min_samples_leaf : int, default=1
        A split is only considered if it leaves at least this many samples
        in *both* children.
    min_impurity_decrease : float, default=0.0
        A split is only made if it decreases impurity by at least this
        much, measured as :math:`\frac{n_t}{n}\Delta H` (weighted by the
        fraction of all training samples in the node, matching
        scikit-learn).

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted unique labels seen in ``y``. ``predict`` returns values from
        here; ``predict_proba`` columns are in this order.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    tree_ : _Node
        Root of the fitted tree.
    max_depth_ : int
        Depth actually reached (0 if the root is a leaf).
    feature_importances_ : ndarray of shape (n_features,)
        Normalised total impurity decrease attributed to each feature
        (the "Gini importance"), summing to 1 — or all zeros if the root is
        a leaf.

    Notes
    -----
    The tree is grown greedily: each node takes the ``(feature,
    threshold)`` split maximising the impurity decrease
    :math:`\Delta H = H(S_t) - \frac{n_L}{n_t}H(S_L)
    - \frac{n_R}{n_t}H(S_R)`, then recurses. Thresholds are midpoints
    between adjacent distinct feature values. Ties are broken
    deterministically (lowest feature index, then lowest threshold), unlike
    scikit-learn which permutes features with its RNG. There is no
    cost-complexity pruning and no feature subsampling. Full walkthrough:
    ``docs/derivations/decision_tree.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.tree import DecisionTreeClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    >>> model.predict(np.array([[3.0], [9.0]])).tolist()
    [0.0, 1.0]
    >>> float(model.tree_.threshold)
    6.0

    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X: FeatureMatrix, y: TargetVector) -> DecisionTreeClassifier:
        """Grow the tree on ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — any number of classes.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``criterion`` is unknown, ``max_depth`` is less than 1,
            ``min_samples_split`` is less than 2, ``min_samples_leaf`` is
            less than 1, or ``min_impurity_decrease`` is negative.

        """
        if self.criterion not in _CRITERIA:
            raise ValueError(
                f"criterion must be one of {tuple(_CRITERIA)}, got {self.criterion!r}."
            )
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")
        if self.min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be >= 2, got {self.min_samples_split}."
            )
        if self.min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}."
            )
        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                f"min_impurity_decrease must be >= 0, got {self.min_impurity_decrease}."
            )

        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]
        y_idx = np.searchsorted(self.classes_, y)  # labels -> 0 .. K-1

        self._impurity_fn = _CRITERIA[self.criterion]
        self._n_total = X.shape[0]
        self._importances = np.zeros(self.n_features_in_)

        self.tree_ = self._build(X, y_idx, depth=0)
        self.max_depth_ = _depth(self.tree_)

        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0.0 else self._importances
        )
        return self

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return class probabilities, shape ``(n_samples, n_classes)``.

        Each row is the class-frequency vector of the leaf the sample falls
        into. Rows sum to 1; columns are ordered as ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        check_is_fitted(self, "tree_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.n_features_in_}."
            )
        # one row at a time down the tree — O(n_queries * depth), clear over fast
        return np.array([_leaf_for(self.tree_, row).value for row in X])

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a class label for each row of ``X``.

        Returns ``classes_[argmax(predict_proba(X))]``; ties resolve to the
        lowest class label (:func:`numpy.argmax` returns the first maximal
        entry and ``classes_`` is sorted).

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        # into a local first — it runs the fitted / feature-count checks, so
        # ``self.classes_`` below can't raise a bare AttributeError pre-fit.
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.accuracy_score`: the fraction of
        samples whose predicted label matches the true label.
        """
        return accuracy_score(y, self.predict(X))

    def _build(self, X: FloatArray, y_idx: IntArray, depth: int) -> _Node:
        """Recursively grow the subtree for the samples ``(X, y_idx)``."""
        n_samples = X.shape[0]
        counts = np.bincount(y_idx, minlength=self.n_classes_).astype(np.float64)
        value = counts / n_samples  # predicted class distribution here
        impurity = float(self._impurity_fn(counts))  # H(S_t)
        node = _Node(n_samples=n_samples, impurity=impurity, value=value)

        at_max_depth = self.max_depth is not None and depth >= self.max_depth
        if at_max_depth or n_samples < self.min_samples_split or impurity == 0.0:
            return node  # leaf

        feature, threshold, gain = _best_split(
            X, y_idx, self.n_classes_, self._impurity_fn, self.min_samples_leaf
        )
        if feature is None:
            return node  # no valid split -> leaf
        # scikit-learn-scaled weighted impurity decrease
        if (n_samples / self._n_total) * gain < self.min_impurity_decrease:
            return node  # split too weak -> leaf

        left_mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y_idx[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y_idx[~left_mask], depth + 1)
        self._importances[feature] += (n_samples / self._n_total) * gain  # N_t/N · ΔH
        return node


def _leaf_for(node: _Node, row: FloatArray) -> _Node:
    """Walk ``row`` from ``node`` down to its leaf."""
    while not node.is_leaf:
        node = node.left if row[node.feature] <= node.threshold else node.right
    return node


def _depth(node: _Node) -> int:
    """Depth of the subtree rooted at ``node`` (a lone leaf has depth 0)."""
    if node.is_leaf:
        return 0
    return 1 + max(_depth(node.left), _depth(node.right))
