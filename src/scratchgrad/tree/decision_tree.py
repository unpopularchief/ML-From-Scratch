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
standard practical substitute. ``max_features`` subsamples the candidate
features at each node (seeded by ``random_state``) — the mechanism a
random forest uses to decorrelate its trees; there is no cost-complexity
pruning (``ccp_alpha``). ``fit`` accepts a ``sample_weight`` vector that
scales each row's contribution to the class counts and the impurity
decrease — the hook the boosting ensembles use to reweight their weak
learner. See ``docs/derivations/decision_tree.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.typing import FeatureMatrix, FloatArray, IntArray, TargetVector
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_sample_weight,
    check_X_y,
)

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


def _resolve_max_features(
    max_features: str | int | float | None, n_features: int
) -> int:
    r"""Resolve the ``max_features`` hyperparameter to a concrete feature count.

    ``None`` -> all ``n_features``; ``"sqrt"`` ->
    :math:`\lfloor\sqrt{n}\rfloor`; ``"log2"`` ->
    :math:`\lfloor\log_2 n\rfloor`; an ``int`` is taken as-is; a ``float``
    is a fraction, :math:`\lfloor f\,n\rfloor`. String and fractional
    results are clamped to at least 1.

    Raises
    ------
    ValueError
        On an unknown string, an ``int`` outside ``[1, n_features]``, or a
        ``float`` outside ``(0, 1]``.

    """
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError(
            f"max_features string must be 'sqrt' or 'log2', got {max_features!r}."
        )
    if isinstance(max_features, bool):
        raise ValueError(f"max_features must not be a bool, got {max_features!r}.")
    if isinstance(max_features, (int, np.integer)):
        if not 1 <= int(max_features) <= n_features:
            raise ValueError(
                f"max_features={int(max_features)} is out of range [1, {n_features}]."
            )
        return int(max_features)
    if isinstance(max_features, float):
        if not 0.0 < max_features <= 1.0:
            raise ValueError(
                f"max_features as a fraction must be in (0, 1], got {max_features}."
            )
        return max(1, int(max_features * n_features))
    raise ValueError(
        "max_features must be None, 'sqrt', 'log2', an int, or a float; "
        f"got {type(max_features).__name__}."
    )


def _best_split(
    X_node: FloatArray,
    y_idx: IntArray,
    n_classes: int,
    impurity_fn: _Criterion,
    min_samples_leaf: int,
    max_features: int | None = None,
    rng: np.random.Generator | None = None,
    sample_weight: FloatArray | None = None,
) -> tuple[int | None, float, float]:
    r"""Best ``(feature, threshold)`` split of a node, and its impurity decrease.

    Sweeps candidate features: sort the node's rows by that feature, let
    the left partition grow one row at a time (its class counts are the
    running cumulative sum of the one-hot labels), and score the split at
    each midpoint between distinct adjacent values. Returns
    ``(None, 0.0, 0.0)`` if no valid split exists (every candidate feature
    constant, or none leaves both children with ``min_samples_leaf`` rows).

    ``rng`` and ``max_features`` drive feature subsampling (see
    ``docs/derivations/decision_tree.md`` §3a). With ``rng`` ``None`` the
    features are swept in index order; otherwise in a fresh
    ``rng.permutation`` order. A feature constant within the node is
    skipped without spending budget; the sweep stops once ``max_features``
    non-constant features have been scored (``None`` = no limit).

    ``sample_weight`` (``None`` = all ones) scales each row: the class
    counts become weighted sums and the child impurities are weighted by
    each side's total weight, so :math:`\Delta H` matches scikit-learn's
    weighted form. The ``min_samples_leaf`` gate still counts *rows*, not
    weight (scikit-learn's convention — the weighted gate is the omitted
    ``min_weight_fraction_leaf``).

    Ties are broken towards the lowest feature index, then the lowest
    threshold — independently of the sweep order, so the result depends on
    the drawn *subset* but not on its permutation.
    """
    n_t = X_node.shape[0]
    n_features = X_node.shape[1]
    if sample_weight is None:
        sample_weight = np.ones(n_t)
    w_total = sample_weight.sum()
    parent_counts = np.bincount(
        y_idx, weights=sample_weight, minlength=n_classes
    ).astype(np.float64)
    parent_impurity = float(impurity_fn(parent_counts))  # H(S_t)
    one_hot = np.eye(n_classes)[y_idx] * sample_weight[:, None]  # weighted, (n_t, K)

    row_left = np.arange(1, n_t)  # left-partition row count at each candidate cut
    row_right = n_t - row_left

    feature_order = range(n_features) if rng is None else rng.permutation(n_features)

    best_gain, best_feature, best_threshold = 0.0, None, 0.0
    n_scored = 0  # non-constant features evaluated so far (the max_features budget)
    for j in feature_order:
        j = int(j)
        order = np.argsort(X_node[:, j], kind="stable")
        x_sorted = X_node[order, j]
        distinct = x_sorted[:-1] != x_sorted[1:]
        if not distinct.any():
            continue  # constant within the node: not a candidate, no budget spent

        left_counts = np.cumsum(one_hot[order], axis=0)[:-1]  # rows order[:i+1]
        right_counts = parent_counts - left_counts  # (n_t - 1, n_classes)
        w_left = left_counts.sum(axis=1)  # total weight on the left at each cut
        w_right = w_total - w_left

        # weighted child impurity at every cut:  (w_L/w_t) H(S_L) + (w_R/w_t) H(S_R)
        child_impurity = (
            w_left * impurity_fn(left_counts) + w_right * impurity_fn(right_counts)
        ) / w_total
        gains = parent_impurity - child_impurity  # ΔH at every cut

        # a split is valid only strictly between two distinct values and only
        # if both children keep at least min_samples_leaf rows
        valid = (
            distinct & (row_left >= min_samples_leaf) & (row_right >= min_samples_leaf)
        )
        if valid.any():
            masked = np.where(valid, gains, -np.inf)
            i = int(np.argmax(masked))  # first maximal cut -> lowest threshold
            gain = float(masked[i])
            improves = gain > best_gain + _MIN_GAIN
            ties_lower_index = (
                best_feature is not None
                and gain > best_gain - _MIN_GAIN
                and j < best_feature
            )
            if gain > _MIN_GAIN and (improves or ties_lower_index):
                best_gain = gain
                best_feature = j
                best_threshold = 0.5 * float(x_sorted[i] + x_sorted[i + 1])  # midpoint

        n_scored += 1
        if max_features is not None and n_scored >= max_features:
            break

    return best_feature, best_threshold, best_gain


@dataclass
class _Node:
    """One node of a fitted tree — internal split or leaf.

    A node is a leaf iff ``left`` is ``None``. ``value`` is the
    class-frequency vector of the training samples that reached it (columns
    ordered as ``classes_``) — or, for ``DecisionTreeRegressor``, the
    scalar weighted mean of their targets; ``impurity`` and ``n_samples``
    are recorded for inspection and for ``feature_importances_``.
    """

    n_samples: int
    impurity: float
    value: FloatArray | float
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
    max_features : {"sqrt", "log2"}, int, float, or None, default=None
        Number of features considered as split candidates at each node.
        ``None`` uses all of them (the fully deterministic tree).
        ``"sqrt"`` / ``"log2"`` use :math:`\lfloor\sqrt d\rfloor` /
        :math:`\lfloor\log_2 d\rfloor`; an int is that count; a float is a
        fraction of ``n_features``. A fresh random subset is drawn at every
        node — the mechanism a random forest uses to decorrelate its
        trees. See ``docs/derivations/decision_tree.md`` §3a.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds the per-node feature subsample. Has no effect when
        ``max_features`` is ``None`` (nothing is drawn) — the tree is then
        fully deterministic and byte-identical to the unseeded fit.

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
    max_features_ : int
        The ``max_features`` hyperparameter resolved to a concrete count.
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
    scikit-learn which permutes features with its RNG. ``max_features``
    (seeded by ``random_state``) subsamples the candidate features per
    node; there is no cost-complexity pruning. :meth:`fit` takes an
    optional ``sample_weight`` — the reweighting hook the boosting
    ensembles use — which scales the class counts and the impurity
    decrease but not the ``min_samples_*`` gates. Full walkthrough:
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
        max_features: str | int | float | None = None,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state

    def fit(
        self,
        X: FeatureMatrix,
        y: TargetVector,
        sample_weight: FloatArray | None = None,
    ) -> DecisionTreeClassifier:
        """Grow the tree on ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — any number of classes.
        sample_weight : ndarray of shape (n_samples,), optional
            Non-negative per-row weights (not all zero). ``None`` (the
            default) weights every row equally, and the fit is then
            byte-identical to calling ``fit(X, y)`` on the pre-weighting
            estimator. Weighted rows scale the class counts and the
            impurity decrease; the ``min_samples_*`` gates still count
            rows, not weight.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``criterion`` is unknown, ``max_depth`` is less than 1,
            ``min_samples_split`` is less than 2, ``min_samples_leaf`` is
            less than 1, ``min_impurity_decrease`` is negative,
            ``max_features`` is not a valid string / an int outside
            ``[1, n_features]`` / a float outside ``(0, 1]``, or
            ``sample_weight`` has the wrong shape / is negative / sums to
            zero.

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
        sample_weight = check_sample_weight(sample_weight, X.shape[0])
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]
        y_idx = np.searchsorted(self.classes_, y)  # labels -> 0 .. K-1

        # Drop zero-weight rows outright (scikit-learn does the same): they
        # contribute nothing to any count, and leaving them in would add
        # spurious candidate split points between real feature values.
        if not np.all(sample_weight > 0.0):
            keep = sample_weight > 0.0
            X, y_idx, sample_weight = X[keep], y_idx[keep], sample_weight[keep]

        self.max_features_ = _resolve_max_features(
            self.max_features, self.n_features_in_
        )
        # No seed and no subsampling -> stay on the deterministic index-order
        # sweep, byte-identical to the M1 tree. Otherwise thread a Generator.
        self._rng = (
            None
            if self.max_features is None and self.random_state is None
            else check_random_state(self.random_state)
        )

        self._impurity_fn = _CRITERIA[self.criterion]
        self._w_total = float(sample_weight.sum())  # total weight (n if uniform)
        self._importances = np.zeros(self.n_features_in_)

        self.tree_ = self._build(X, y_idx, sample_weight, depth=0)
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

    def _build(
        self, X: FloatArray, y_idx: IntArray, w: FloatArray, depth: int
    ) -> _Node:
        """Recursively grow the subtree for the samples ``(X, y_idx)``.

        ``w`` is the sample-weight slice for this node's rows; with uniform
        weights it is all ones and every quantity below reduces to the
        row-count version.
        """
        n_samples = X.shape[0]
        counts = np.bincount(y_idx, weights=w, minlength=self.n_classes_).astype(
            np.float64
        )
        w_node = counts.sum()  # total weight reaching this node (n_samples if uniform)
        value = counts / w_node  # predicted class distribution here
        impurity = float(self._impurity_fn(counts))  # H(S_t)
        node = _Node(n_samples=n_samples, impurity=impurity, value=value)

        at_max_depth = self.max_depth is not None and depth >= self.max_depth
        if at_max_depth or n_samples < self.min_samples_split or impurity == 0.0:
            return node  # leaf

        feature, threshold, gain = _best_split(
            X,
            y_idx,
            self.n_classes_,
            self._impurity_fn,
            self.min_samples_leaf,
            self.max_features_,
            self._rng,
            w,
        )
        if feature is None:
            return node  # no valid split -> leaf
        # scikit-learn-scaled weighted impurity decrease:  (N_t / N) · ΔH
        if (w_node / self._w_total) * gain < self.min_impurity_decrease:
            return node  # split too weak -> leaf

        left_mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y_idx[left_mask], w[left_mask], depth + 1)
        node.right = self._build(
            X[~left_mask], y_idx[~left_mask], w[~left_mask], depth + 1
        )
        self._importances[feature] += (w_node / self._w_total) * gain  # (N_t/N) · ΔH
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
