r"""CART decision tree — regression.

The regression companion to :mod:`scratchgrad.tree.decision_tree`. The
*search* is identical — greedy recursive binary partitioning, every
internal node testing ``x[feature] <= threshold``, midpoint thresholds,
the deterministic lowest-index tie-break, ``max_features`` / ``random_state``
per-node subsampling, and a ``sample_weight`` hook. Only three things
change:

* **impurity** is the within-node *variance* of ``y`` rather than a class
  impurity — a split maximises the variance reduction

  .. math::
      \Delta H(j, \tau) = H(S_t)
        - \frac{N_L}{N_t} H(S_L) - \frac{N_R}{N_t} H(S_R),
      \qquad
      H(t) = \frac{1}{N_t}\sum_{i \in S_t} w_i (y_i - \bar y_t)^2,

  which (law of total variance) is the between-group variance of the split
  and equals the greedy drop in training sum-of-squared-error;
* each **leaf** predicts the single scalar :math:`\bar y_t`, the weighted
  mean of its training targets (the constant that minimises the node SSE);
* **prediction** returns that scalar and **score** is the :math:`R^2`
  coefficient of determination.

Only ``criterion="squared_error"`` is implemented; ``absolute_error`` /
``friedman_mse`` / ``poisson`` are out of scope. See
``docs/derivations/decision_tree_regressor.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.regression import r2_score
from scratchgrad.tree.decision_tree import (
    _MIN_GAIN,
    _depth,
    _leaf_for,
    _Node,
    _resolve_max_features,
)
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_sample_weight,
    check_X_y,
)

_CRITERIA = ("squared_error",)


def _variance_from_sums(
    w: FloatArray | float, s: FloatArray | float, q: FloatArray | float
) -> FloatArray:
    r"""Weighted variance :math:`\frac{q}{w} - (s/w)^2` from its moment sums.

    ``w`` is the total weight, ``s = sum w_i y_i``, ``q = sum w_i y_i^2``.
    Broadcasts elementwise, so scalars return a scalar and equal-shaped
    arrays return one variance per position — the form the split sweep
    needs to score every candidate cut at once. Catastrophic cancellation
    can push the result a few ulps below zero on a near-constant node, so
    it is clamped at ``0``. A zero-weight position returns ``0``.
    """
    w_safe = np.where(np.asarray(w) == 0.0, 1.0, w)
    var = q / w_safe - (s / w_safe) ** 2
    return np.maximum(var, 0.0)


def _best_split(
    X_node: FloatArray,
    y: FloatArray,
    min_samples_leaf: int,
    max_features: int | None = None,
    rng: np.random.Generator | None = None,
    sample_weight: FloatArray | None = None,
) -> tuple[int | None, float, float]:
    r"""Best ``(feature, threshold)`` split of a node, and its variance decrease.

    The regression analogue of
    :func:`scratchgrad.tree.decision_tree._best_split`: sweep candidate
    features, sort the node's rows by that feature, let the left partition
    grow one row at a time, and score the split at each midpoint between
    distinct adjacent values. Where the classifier accumulates a cumulative
    sum of one-hot labels, this accumulates three scalar running sums per
    cut — :math:`W_L = \sum w_i`, :math:`P_L = \sum w_i y_i`,
    :math:`Q_L = \sum w_i y_i^2` — and turns them into each side's variance
    via :func:`_variance_from_sums`. Returns ``(None, 0.0, 0.0)`` if no
    valid split exists.

    ``rng`` / ``max_features`` drive feature subsampling and
    ``sample_weight`` (``None`` = all ones) scales every sum, both exactly
    as in the classifier (see
    ``docs/derivations/decision_tree_regressor.md`` §4a). The
    ``min_samples_leaf`` gate counts *rows*. Ties break towards the lowest
    feature index, then the lowest threshold, independently of the sweep
    order — with "tie" and "too small to count" measured *relative to the
    node's impurity*, so the tree does not depend on the units of ``y``.
    """
    n_t, n_features = X_node.shape
    if sample_weight is None:
        sample_weight = np.ones(n_t)
    w = sample_weight
    w_total = w.sum()

    # Centre y on the node mean before accumulating any moment sums. Variance
    # — and therefore every gain below — is exactly shift-invariant, so this
    # changes nothing mathematically, but it is what makes the sum form
    # `q/w - (s/w)^2` usable: that expression subtracts two nearly equal
    # numbers, and when |mean| >> spread (a target with a large baseline —
    # timestamps, prices, Kelvin) the cancellation eats every significant
    # digit. Uncentred, a spread-1 target offset by 1e9 has its variance come
    # out as exactly 0.0, so the node looks pure and the tree stops splitting.
    # One O(n_t) pass buys back the precision.
    y = y - (w @ y) / w_total

    wy = w * y
    wy2 = wy * y
    parent_s = wy.sum()
    parent_q = wy2.sum()
    parent_impurity = float(_variance_from_sums(w_total, parent_s, parent_q))  # H(S_t)

    # `_MIN_GAIN` is an *absolute* floor in the classifier, where a gini or
    # entropy decrease always lies in [0, 1] and 1e-12 is genuinely "this is
    # float noise". A variance decrease instead carries the units of y^2:
    # rescaling y by s scales every gain by s^2, so the same constant would
    # reject perfectly real splits just because y is small (targets in
    # millivolts, say) and silently return a stump. Scale the floor by the
    # node's own impurity — gain <= parent_impurity always, so this asks
    # "is the gain a negligible *fraction* of this node's variance?", which
    # is what the guard is actually for and makes the fitted tree invariant
    # to the units of y.
    gain_atol = _MIN_GAIN * parent_impurity if parent_impurity > 0.0 else _MIN_GAIN

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

        w_left = np.cumsum(w[order])[:-1]  # sums over rows order[:i+1]
        s_left = np.cumsum(wy[order])[:-1]
        q_left = np.cumsum(wy2[order])[:-1]
        w_right = w_total - w_left
        s_right = parent_s - s_left
        q_right = parent_q - q_left

        # weighted child impurity at every cut:  (W_L/W) H(S_L) + (W_R/W) H(S_R)
        child_impurity = (
            w_left * _variance_from_sums(w_left, s_left, q_left)
            + w_right * _variance_from_sums(w_right, s_right, q_right)
        ) / w_total
        gains = parent_impurity - child_impurity  # ΔH at every cut

        valid = (
            distinct & (row_left >= min_samples_leaf) & (row_right >= min_samples_leaf)
        )
        if valid.any():
            masked = np.where(valid, gains, -np.inf)
            i = int(np.argmax(masked))  # first maximal cut -> lowest threshold
            gain = float(masked[i])
            improves = gain > best_gain + gain_atol
            ties_lower_index = (
                best_feature is not None
                and gain > best_gain - gain_atol
                and j < best_feature
            )
            if gain > gain_atol and (improves or ties_lower_index):
                best_gain = gain
                best_feature = j
                best_threshold = 0.5 * float(x_sorted[i] + x_sorted[i + 1])  # midpoint

        n_scored += 1
        if max_features is not None and n_scored >= max_features:
            break

    return best_feature, best_threshold, best_gain


class DecisionTreeRegressor(Estimator):
    r"""A CART decision tree for regression.

    Parameters
    ----------
    criterion : {"squared_error"}, default="squared_error"
        Split-quality measure. Only the mean-squared-error criterion (the
        within-node variance of ``y``) is implemented;
        ``"absolute_error"`` / ``"friedman_mse"`` / ``"poisson"`` are out
        of scope.
    max_depth : int, optional
        Maximum depth of the tree. ``None`` (the default) grows nodes until
        they are pure (constant ``y``) or too small to split. The root
        alone has depth 0.
    min_samples_split : int, default=2
        A node with fewer rows than this becomes a leaf.
    min_samples_leaf : int, default=1
        A split is only considered if it leaves at least this many rows in
        *both* children.
    min_impurity_decrease : float, default=0.0
        A split is only made if it decreases impurity by at least this
        much, measured as :math:`\frac{N_t}{N}\Delta H` (weighted by the
        node's fraction of the total training mass, matching scikit-learn).
    max_features : {"sqrt", "log2"}, int, float, or None, default=None
        Number of features considered as split candidates at each node.
        ``None`` uses all of them (the fully deterministic tree). ``"sqrt"``
        / ``"log2"`` use :math:`\lfloor\sqrt d\rfloor` /
        :math:`\lfloor\log_2 d\rfloor`; an int is that count; a float is a
        fraction of ``n_features``. A fresh random subset is drawn at every
        node. See ``docs/derivations/decision_tree.md`` §3a.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds the per-node feature subsample. Has no effect when
        ``max_features`` is ``None`` — the tree is then fully deterministic
        and byte-identical to the unseeded fit.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    tree_ : _Node
        Root of the fitted tree. Each leaf's ``value`` is the scalar
        weighted mean of the training targets that reached it.
    max_depth_ : int
        Depth actually reached (0 if the root is a leaf).
    max_features_ : int
        The ``max_features`` hyperparameter resolved to a concrete count.
    feature_importances_ : ndarray of shape (n_features,)
        Normalised total variance decrease attributed to each feature,
        summing to 1 — or all zeros if the root is a leaf.

    Notes
    -----
    The tree is grown greedily: each node takes the ``(feature,
    threshold)`` split maximising the variance decrease
    :math:`\Delta H = H(S_t) - \frac{N_L}{N_t}H(S_L)
    - \frac{N_R}{N_t}H(S_R)` with :math:`H` the weighted variance of
    ``y``, then recurses. By the law of total variance this is the
    between-group variance of the split, i.e. the greedy drop in training
    SSE. Thresholds are midpoints between adjacent distinct feature values;
    ties break deterministically (lowest feature index, then lowest
    threshold). :meth:`fit` takes an optional ``sample_weight``. Full
    walkthrough: ``docs/derivations/decision_tree_regressor.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.tree import DecisionTreeRegressor
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    >>> y = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    >>> model = DecisionTreeRegressor(max_depth=1).fit(X, y)
    >>> model.predict(np.array([[3.0], [9.0]])).tolist()
    [1.0, 9.0]
    >>> float(model.tree_.threshold)
    6.0

    """

    def __init__(
        self,
        criterion: str = "squared_error",
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
    ) -> DecisionTreeRegressor:
        """Grow the tree on ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Continuous training targets.
        sample_weight : ndarray of shape (n_samples,), optional
            Non-negative per-row weights (not all zero). ``None`` (the
            default) weights every row equally. Weighted rows scale the
            moment sums and the variance decrease; the ``min_samples_*``
            gates still count rows, not weight. Zero-weight rows are
            dropped before fitting, as scikit-learn does.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``criterion`` is not ``"squared_error"``, ``max_depth`` is
            less than 1, ``min_samples_split`` is less than 2,
            ``min_samples_leaf`` is less than 1, ``min_impurity_decrease``
            is negative, ``max_features`` is not a valid string / an int
            outside ``[1, n_features]`` / a float outside ``(0, 1]``, or
            ``sample_weight`` has the wrong shape / is negative / sums to
            zero.

        """
        if self.criterion not in _CRITERIA:
            raise ValueError(
                f"criterion must be one of {_CRITERIA}, got {self.criterion!r}."
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
        self.n_features_in_ = X.shape[1]

        # Drop zero-weight rows outright (scikit-learn does the same): they
        # contribute nothing to any sum, and leaving them in would add
        # spurious candidate split points between real feature values.
        if not np.all(sample_weight > 0.0):
            keep = sample_weight > 0.0
            X, y, sample_weight = X[keep], y[keep], sample_weight[keep]

        self.max_features_ = _resolve_max_features(
            self.max_features, self.n_features_in_
        )
        # No seed and no subsampling -> deterministic index-order sweep.
        self._rng = (
            None
            if self.max_features is None and self.random_state is None
            else check_random_state(self.random_state)
        )

        self._w_total = float(sample_weight.sum())  # total weight (n if uniform)
        self._importances = np.zeros(self.n_features_in_)

        self.tree_ = self._build(X, y, sample_weight, depth=0)
        self.max_depth_ = _depth(self.tree_)

        total = self._importances.sum()
        self.feature_importances_ = (
            self._importances / total if total > 0.0 else self._importances
        )
        return self

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a target value for each row of ``X``.

        Each prediction is the scalar ``value`` of the leaf the sample
        falls into — the weighted mean of the training targets that
        reached it.

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
        return np.array(
            [_leaf_for(self.tree_, row).value for row in X], dtype=np.float64
        )

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        r"""Return the :math:`R^2` of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.r2_score`. 1.0 is a perfect fit,
        0.0 matches a constant "predict the mean" baseline, negative is
        worse than that baseline.
        """
        return r2_score(y, self.predict(X))

    def _build(self, X: FloatArray, y: FloatArray, w: FloatArray, depth: int) -> _Node:
        """Recursively grow the subtree for the samples ``(X, y)``.

        ``w`` is the sample-weight slice for this node's rows; with uniform
        weights it is all ones and every quantity below reduces to the
        unweighted version.
        """
        n_samples = X.shape[0]
        w_node = w.sum()  # total weight reaching this node (n_samples if uniform)
        mean = float(w @ y / w_node)  # predicted value here (SSE-minimising constant)
        # centred form -> exactly 0.0 on a constant-y node, so the pure-node
        # check below is clean (the sum form can leave a few-ulp residual)
        impurity = float(w @ (y - mean) ** 2 / w_node)  # H(S_t)
        node = _Node(n_samples=n_samples, impurity=impurity, value=mean)

        at_max_depth = self.max_depth is not None and depth >= self.max_depth
        if at_max_depth or n_samples < self.min_samples_split or impurity == 0.0:
            return node  # leaf

        feature, threshold, gain = _best_split(
            X,
            y,
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
        node.left = self._build(X[left_mask], y[left_mask], w[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], w[~left_mask], depth + 1)
        self._importances[feature] += (w_node / self._w_total) * gain  # (N_t/N) · ΔH
        return node
