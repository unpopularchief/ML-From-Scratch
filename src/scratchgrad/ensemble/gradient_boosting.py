r"""Gradient boosting — classification (log-loss / deviance).

Gradient descent in function space (Friedman, 2001). The model keeps a raw
additive score :math:`F(x)` (a log-odds); each round fits a
``DecisionTreeRegressor`` to the **negative gradient** of the loss w.r.t.
that score — the pseudo-residual :math:`y_i - p_i` for the binomial
deviance — then replaces the tree's leaf values with a one-step Newton
estimate of the loss-minimising constant on each leaf ("TreeBoost"):

.. math::
    \gamma_{j} = \frac{\sum_{i \in R_j} w_i\,(y_i - p_i)}
                      {\sum_{i \in R_j} w_i\,p_i(1 - p_i)} .

The scores are updated by :math:`F \mathrel{+}= \nu\,\gamma`, with
:math:`\nu` the ``learning_rate``. Multiclass fits :math:`K` trees per
round against the multinomial-deviance residuals :math:`Y_{ik} - p_{ik}`,
with an extra :math:`\frac{K-1}{K}` factor on the leaf update.
``subsample < 1`` fits each tree on a fresh row subsample (stochastic
gradient boosting, Friedman 2002).

Log-loss only — ``loss="exponential"`` (which would reproduce AdaBoost) and
``GradientBoostingRegressor`` are out of scope. Full walkthrough:
``docs/derivations/gradient_boosting.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.tree import DecisionTreeRegressor
from scratchgrad.tree.decision_tree import _leaf_for, _Node
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.math import sigmoid, softmax
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_sample_weight,
    check_X_y,
)

# Probabilities are clipped this far from {0, 1} before the init log-odds /
# log-priors, so a class that is absent or ubiquitous in the training set
# does not send the initial score to +-inf.
_PROBA_EPS = 1e-12
# A Newton leaf denominator below this is treated as zero: the whole leaf
# has near-certain predictions, the numerator has collapsed too, and the
# ratio is 0/0. scikit-learn makes the same guard.
_MIN_HESSIAN = 1e-12


def _log_odds(y01: FloatArray, sample_weight: FloatArray) -> float:
    r"""Weighted-base-rate log-odds :math:`\log\frac{\bar y}{1 - \bar y}`.

    The constant score that minimises the binomial deviance — the
    ``init="prior"`` starting point for the binary path. ``y01`` is the
    ``{0, 1}`` indicator of the positive class.
    """
    p = float(np.average(y01, weights=sample_weight))
    p = min(max(p, _PROBA_EPS), 1.0 - _PROBA_EPS)  # keep the log finite
    return float(np.log(p / (1.0 - p)))


def _prior_logits(y_onehot: FloatArray, sample_weight: FloatArray) -> FloatArray:
    r"""Per-class log-priors :math:`\log \pi_k`, shape ``(n_classes,)``.

    ``softmax`` of this vector recovers the weighted class frequencies
    :math:`\pi_k` — the ``init="prior"`` starting scores for the multiclass
    path.
    """
    priors = (sample_weight @ y_onehot) / sample_weight.sum()  # pi_k
    priors = np.clip(priors, _PROBA_EPS, 1.0 - _PROBA_EPS)
    return np.log(priors)


def _binary_negative_gradient(y01: FloatArray, raw: FloatArray) -> FloatArray:
    r"""Pseudo-residual :math:`y_i - \sigma(F_i)` for the binomial deviance."""
    return y01 - sigmoid(raw)


def _multinomial_negative_gradient(y_onehot: FloatArray, raw: FloatArray) -> FloatArray:
    r"""Pseudo-residual :math:`Y_{ik} - \mathrm{softmax}(F_i)_k`, shape ``(n, K)``."""
    return y_onehot - softmax(raw, axis=1)


def _pseudo_hessian(negative_gradient: FloatArray) -> FloatArray:
    r"""Diagonal Hessian :math:`|r|\,(1 - |r|)` of the deviance from its gradient.

    For one-hot targets :math:`r = Y_k - p_k`, so :math:`|r| = 1 - p_k`
    when :math:`Y_k = 1` and :math:`|r| = p_k` when :math:`Y_k = 0`; either
    way :math:`|r|(1 - |r|) = p_k(1 - p_k)`, the second derivative the
    Newton leaf update needs. Friedman's form, and scikit-learn's.
    """
    g = np.abs(negative_gradient)
    return g * (1.0 - g)


def _newton_leaf_value(
    negative_gradient: FloatArray,
    hessian: FloatArray,
    sample_weight: FloatArray,
    factor: float,
) -> float:
    r"""One-step Newton estimate of a leaf's loss-minimising value.

    :math:`\text{factor} \cdot \dfrac{\sum_i w_i\,r_i}{\sum_i w_i\,h_i}`,
    with ``factor`` 1 for the binomial deviance and
    :math:`\frac{K-1}{K}` for the multinomial one. Returns ``0.0`` when the
    denominator underflows (a leaf whose samples are already near-certain).
    """
    numerator = float(np.sum(sample_weight * negative_gradient))
    denominator = float(np.sum(sample_weight * hessian))
    if denominator < _MIN_HESSIAN:
        return 0.0
    return factor * numerator / denominator


def _refine_leaves(
    tree: DecisionTreeRegressor,
    X_in_bag: FloatArray,
    negative_gradient: FloatArray,
    hessian: FloatArray,
    sample_weight: FloatArray,
    factor: float,
) -> None:
    """Overwrite each leaf ``value`` of a freshly fit tree with its Newton estimate.

    The tree was fit to the pseudo-residuals under ``squared_error``, so
    its leaves hold mean residuals; TreeBoost keeps the *partition* but
    replaces those with the constant that best reduces the true loss on the
    leaf's in-bag rows (``docs/derivations/gradient_boosting.md`` §2).
    Mutates ``tree.tree_`` in place — the same thing scikit-learn does.
    """
    buckets: dict[int, tuple[_Node, list[int]]] = {}
    for i, row in enumerate(X_in_bag):
        leaf = _leaf_for(tree.tree_, row)
        buckets.setdefault(id(leaf), (leaf, []))[1].append(i)
    for leaf, members in buckets.values():
        rows = np.asarray(members)
        leaf.value = _newton_leaf_value(
            negative_gradient[rows], hessian[rows], sample_weight[rows], factor
        )


def _binary_deviance(y01: FloatArray, raw: FloatArray, w: FloatArray) -> float:
    r"""Weighted mean binomial deviance :math:`\mathrm{softplus}(F) - yF`."""
    loss = np.logaddexp(0.0, raw) - y01 * raw  # softplus(F) - y F
    return float(np.average(loss, weights=w))


def _multinomial_deviance(
    y_onehot: FloatArray, raw: FloatArray, w: FloatArray
) -> float:
    r"""Weighted mean multinomial deviance :math:`\log\sum_k e^{F_k} - F_{y}`."""
    log_norm = np.log(np.sum(np.exp(raw - raw.max(axis=1, keepdims=True)), axis=1))
    log_norm += raw.max(axis=1)  # stable logsumexp along the class axis
    loss = log_norm - np.sum(y_onehot * raw, axis=1)
    return float(np.average(loss, weights=w))


class GradientBoostingClassifier(Estimator):
    r"""Gradient boosting for classification (log-loss / deviance).

    Parameters
    ----------
    learning_rate : float, default=0.1
        Shrinkage :math:`\nu` applied to every leaf update. Smaller values
        need more estimators but generalise better. Must be positive.
    n_estimators : int, default=100
        Number of boosting rounds. For a :math:`K`-class problem with
        :math:`K > 2` each round fits :math:`K` trees, so the total tree
        count is ``n_estimators * K``.
    subsample : float in (0, 1], default=1.0
        Fraction of rows drawn without replacement to fit each tree
        (stochastic gradient boosting). ``1.0`` is deterministic ordinary
        boosting; ``< 1.0`` needs ``random_state`` for reproducibility.
    max_depth : int, default=3
        Depth of each base ``DecisionTreeRegressor``. Must be at least 1.
    min_samples_split : int, default=2
        Passed to each base tree.
    min_samples_leaf : int, default=1
        Passed to each base tree.
    min_impurity_decrease : float, default=0.0
        Passed to each base tree.
    max_features : {"sqrt", "log2"}, int, float, or None, default=None
        Per-node feature subsampling for each base tree. ``None`` considers
        every feature (the deterministic path). See
        ``docs/derivations/decision_tree.md`` §3a.
    init : {"prior", "zero"}, default="prior"
        Starting raw score. ``"prior"`` uses the weighted base-rate
        log-odds (binary) or per-class log-priors (multiclass) — the
        constant that minimises the loss. ``"zero"`` starts at 0.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds the ``subsample`` draw and, when ``max_features`` is set, the
        per-node feature draw of every base tree. Has no effect on the
        fully deterministic path (``subsample=1.0``, ``max_features=None``).

    Attributes
    ----------
    estimators_ : list of list of DecisionTreeRegressor
        ``estimators_[m]`` holds the tree(s) fit in round ``m`` — one for a
        binary problem, ``n_classes_`` for a multiclass one.
    classes_ : ndarray of shape (n_classes,)
        Sorted unique labels. ``predict`` returns values from here;
        ``predict_proba`` / ``decision_function`` columns are in this
        order.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    init_score_ : ndarray
        The raw score every sample starts from — shape ``(1,)`` for a
        binary problem, ``(n_classes_,)`` for a multiclass one.
    train_score_ : ndarray of shape (n_estimators,)
        Weighted-mean training loss (deviance) after each round.
    feature_importances_ : ndarray of shape (n_features,)
        Mean of the per-tree variance-decrease importances over every
        fitted tree, renormalised to sum to 1 (all zeros if no tree ever
        splits).

    Notes
    -----
    Deviance boosting targets the log-odds directly, so ``predict_proba``
    is a genuine posterior estimate (unlike AdaBoost's monotone score
    calibration). Rounds are sequential by nature — there is no ``n_jobs``.
    No exact scikit-learn parity: the base tree uses ``squared_error`` and
    a deterministic tie-break where ``GradientBoostingClassifier`` uses
    ``friedman_mse`` and an RNG-permuted split search. ``loss="exponential"``
    and ``GradientBoostingRegressor`` are out of scope. Full walkthrough:
    ``docs/derivations/gradient_boosting.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.ensemble import GradientBoostingClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
    >>> model.predict(np.array([[3.0], [9.0]])).tolist()
    [0.0, 1.0]
    >>> model.predict_proba(np.array([[3.0]])).shape
    (1, 2)

    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        max_features: str | int | float | None = None,
        init: str = "prior",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.init = init
        self.random_state = random_state

    def fit(
        self,
        X: FeatureMatrix,
        y: TargetVector,
        sample_weight: FloatArray | None = None,
    ) -> GradientBoostingClassifier:
        """Run the gradient-boosting loop on ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — two or more classes.
        sample_weight : ndarray of shape (n_samples,), optional
            Non-negative per-row weights (not all zero). ``None`` weights
            every row equally. Weights enter the init score, the Newton
            leaf updates, the base-tree fits, and ``train_score_``.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``n_estimators < 1``, ``learning_rate <= 0``, ``subsample``
            is outside ``(0, 1]``, ``max_depth < 1``, ``init`` is unknown,
            ``y`` has fewer than two classes, ``sample_weight`` is
            malformed, or a base-tree hyperparameter is invalid.

        """
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError(f"subsample must be in (0, 1], got {self.subsample}.")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")
        if self.init not in ("prior", "zero"):
            raise ValueError(f"init must be 'prior' or 'zero', got {self.init!r}.")

        X, y = check_X_y(X, y)
        w = check_sample_weight(sample_weight, X.shape[0])
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        if self.n_classes_ < 2:
            raise ValueError(f"y needs at least 2 classes, got {self.n_classes_}.")
        self.n_features_in_ = X.shape[1]
        n = X.shape[0]

        binary = self.n_classes_ == 2
        self._n_scores = 1 if binary else self.n_classes_
        factor = 1.0 if binary else (self.n_classes_ - 1) / self.n_classes_

        # one-hot targets, kept as (n, n_scores) so binary and multiclass
        # share the loop below (binary carries the positive-class column only)
        if binary:
            y_onehot = (y == self.classes_[1]).astype(np.float64)[:, None]
        else:
            y_onehot = (y[:, None] == self.classes_[None, :]).astype(np.float64)

        if self.init == "zero":
            self.init_score_ = np.zeros(self._n_scores)
        elif binary:
            self.init_score_ = np.array([_log_odds(y_onehot[:, 0], w)])
        else:
            self.init_score_ = _prior_logits(y_onehot, w)
        raw = np.tile(self.init_score_, (n, 1))  # F^(0), shape (n, n_scores)

        needs_rng = (
            self.random_state is not None
            or self.subsample < 1.0
            or self.max_features is not None
        )
        self._rng = check_random_state(self.random_state) if needs_rng else None
        # a base tree only consumes randomness when it subsamples features
        tree_random_state = self._rng if self.max_features is not None else None
        n_sub = max(1, int(round(self.subsample * n)))

        self.estimators_ = []
        train_score: list[float] = []
        for _ in range(self.n_estimators):
            if binary:
                residual = _binary_negative_gradient(y_onehot[:, 0], raw[:, 0])[:, None]
            else:
                residual = _multinomial_negative_gradient(y_onehot, raw)
            hessian = _pseudo_hessian(residual)

            if self.subsample < 1.0:
                in_bag = self._rng.choice(n, size=n_sub, replace=False)
            else:
                in_bag = np.arange(n)

            round_trees: list[DecisionTreeRegressor] = []
            for k in range(self._n_scores):
                tree = DecisionTreeRegressor(
                    criterion="squared_error",
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_impurity_decrease=self.min_impurity_decrease,
                    max_features=self.max_features,
                    random_state=tree_random_state,
                )
                tree.fit(X[in_bag], residual[in_bag, k], sample_weight=w[in_bag])
                _refine_leaves(
                    tree,
                    X[in_bag],
                    residual[in_bag, k],
                    hessian[in_bag, k],
                    w[in_bag],
                    factor,
                )
                raw[:, k] += self.learning_rate * tree.predict(X)
                round_trees.append(tree)
            self.estimators_.append(round_trees)

            if binary:
                train_score.append(_binary_deviance(y_onehot[:, 0], raw[:, 0], w))
            else:
                train_score.append(_multinomial_deviance(y_onehot, raw, w))

        self.train_score_ = np.array(train_score)

        all_trees = [tree for round_trees in self.estimators_ for tree in round_trees]
        importances = np.mean([tree.feature_importances_ for tree in all_trees], axis=0)
        total = importances.sum()
        self.feature_importances_ = importances / total if total > 0.0 else importances
        return self

    def _check_predict_input(self, X: FeatureMatrix) -> FloatArray:
        """Run the fitted / array / feature-count checks, return the coerced ``X``."""
        check_is_fitted(self, "estimators_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.n_features_in_}."
            )
        return X

    def _raw_predict(self, X: FeatureMatrix) -> FloatArray:
        """Return the additive raw score ``F(X)``, shape ``(n_samples, n_scores)``."""
        X = self._check_predict_input(X)
        raw = np.tile(self.init_score_, (X.shape[0], 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                raw[:, k] += self.learning_rate * tree.predict(X)
        return raw

    def _staged_raw(self, X: FeatureMatrix) -> Iterator[FloatArray]:
        """Yield the raw score after each round; the array is reused between yields."""
        X = self._check_predict_input(X)
        raw = np.tile(self.init_score_, (X.shape[0], 1))
        for round_trees in self.estimators_:
            for k, tree in enumerate(round_trees):
                raw[:, k] += self.learning_rate * tree.predict(X)
            yield raw

    def _proba_from_raw(self, raw: FloatArray) -> FloatArray:
        """Map a raw score ``(n, n_scores)`` to a full ``(n, n_classes)`` posterior."""
        if self.n_classes_ == 2:
            p = sigmoid(raw[:, 0])
            return np.column_stack([1.0 - p, p])
        return softmax(raw, axis=1)

    def decision_function(self, X: FeatureMatrix) -> FloatArray:
        """Return the additive raw score.

        Shape ``(n_samples,)`` for a binary problem (the single score,
        positive favours ``classes_[1]``); ``(n_samples, n_classes)`` for a
        multiclass one.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        raw = self._raw_predict(X)
        return raw[:, 0] if self.n_classes_ == 2 else raw

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return class posteriors, shape ``(n_samples, n_classes)``.

        ``sigmoid`` of the raw score for a binary problem (columns
        ``[1 - p, p]``), ``softmax`` for a multiclass one. Rows sum to 1;
        columns follow ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        return self._proba_from_raw(self._raw_predict(X))

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a class label for each row of ``X``.

        ``classes_[argmax(predict_proba)]`` — threshold at 0.5 for a binary
        problem — with ties resolving to the lowest label.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def staged_decision_function(self, X: FeatureMatrix) -> Iterator[FloatArray]:
        """Yield :meth:`decision_function` for the ensemble after each round.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        for raw in self._staged_raw(X):
            yield raw[:, 0].copy() if self.n_classes_ == 2 else raw.copy()

    def staged_predict_proba(self, X: FeatureMatrix) -> Iterator[FloatArray]:
        """Yield :meth:`predict_proba` for the ensemble after each round."""
        for raw in self._staged_raw(X):
            yield self._proba_from_raw(raw)

    def staged_predict(self, X: FeatureMatrix) -> Iterator[TargetVector]:
        """Yield :meth:`predict` for the ensemble after each round.

        The sequence that traces the boosting curve.
        """
        for proba in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``."""
        return accuracy_score(y, self.predict(X))
