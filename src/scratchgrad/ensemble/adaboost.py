r"""AdaBoost — classification (SAMME).

Forward stagewise additive modelling of the multi-class exponential loss
:math:`\exp(-\tfrac{1}{K}\,\mathbf{y}^\top \mathbf{f})` with a symmetric
class encoding (Zhu, Zou, Rosset & Hastie, 2009). Each round fits a
``DecisionTreeClassifier`` (a depth-1 stump by default) on the training
rows reweighted by the previous round's mistakes, then adds it to the vote
with weight

.. math::
    \alpha_m = \log\frac{1 - \mathrm{err}_m}{\mathrm{err}_m} + \log(K - 1),

where :math:`\mathrm{err}_m` is the stump's weighted error. Misclassified
rows have their weight multiplied by :math:`e^{\eta\alpha_m}` before the
next round, so boosting concentrates on the hard cases. Prediction is the
:math:`\alpha`-weighted vote; for :math:`K = 2` the whole scheme reduces to
Freund & Schapire's original discrete AdaBoost.

This is SAMME — scikit-learn's only boosting algorithm since 1.6 (the
``SAMME.R`` variant was removed). ``AdaBoostRegressor`` is a different
algorithm and is not implemented. Full walkthrough:
``docs/derivations/adaboost.md``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.tree import DecisionTreeClassifier
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.math import softmax
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_sample_weight,
    check_X_y,
)


def _weighted_error(
    y_true: FloatArray, y_pred: FloatArray, sample_weight: FloatArray
) -> float:
    r"""Return the weighted misclassification rate of ``y_pred``.

    :math:`\sum_i w_i\,\mathbb{1}[y_i \ne \hat y_i]`. ``sample_weight`` is
    assumed already normalised to sum to 1, so the result is in ``[0, 1]``.
    """
    return float(sample_weight[y_true != y_pred].sum())


def _samme_alpha(err: float, n_classes: int, learning_rate: float) -> float:
    r"""Return the SAMME estimator weight :math:`\eta(\log\frac{1-e}{e} + \log(K-1))`.

    For ``n_classes == 2`` the :math:`\log(K-1)` term is zero and this is
    the classic AdaBoost weight. Only defined for
    ``0 < err < 1 - 1 / n_classes`` (a weak learner that beats random
    guessing); the caller handles the degenerate cases.
    """
    return float(learning_rate * (np.log((1.0 - err) / err) + np.log(n_classes - 1)))


def _samme_decision(
    estimators: list[DecisionTreeClassifier],
    weights: FloatArray,
    classes: FloatArray,
    X: FloatArray,
) -> FloatArray:
    r"""Return the SAMME aggregated score, shape ``(n_samples, n_classes)``.

    Each stump contributes ``+w`` to the column of the class it predicts
    and ``-w / (K - 1)`` to every other column (so its contribution sums to
    zero across classes), and the total is divided by ``weights.sum()``.
    This matches scikit-learn's ``decision_function`` up to the binary
    collapse; ``argmax`` over it is the prediction.
    """
    n_samples, n_classes = X.shape[0], classes.shape[0]
    score = np.zeros((n_samples, n_classes))
    for tree, w in zip(estimators, weights, strict=True):
        pred_idx = np.searchsorted(classes, tree.predict(X))  # predicted class column
        contribution = np.full((n_samples, n_classes), -1.0 / (n_classes - 1))
        contribution[np.arange(n_samples), pred_idx] = 1.0
        score += w * contribution
    return score / weights.sum()


class AdaBoostClassifier(Estimator):
    r"""AdaBoost classifier (SAMME) on decision-tree weak learners.

    Parameters
    ----------
    n_estimators : int, default=50
        Maximum number of boosting rounds. Fewer are kept if a round fits
        the reweighted data perfectly or a round's weak learner fails to
        beat random guessing.
    learning_rate : float, default=1.0
        Shrinkage :math:`\eta` applied to every estimator weight
        :math:`\alpha_m` (and hence to the weight update). Smaller values
        need more estimators. Must be positive.
    max_depth : int, default=1
        Depth of each base ``DecisionTreeClassifier``. ``1`` is a decision
        stump — the canonical AdaBoost weak learner; larger values boost
        shallow trees.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The fitted weak learners, in boosting order.
    estimator_weights_ : ndarray of shape (n_rounds,)
        The vote weight :math:`\alpha_m` of each estimator.
    estimator_errors_ : ndarray of shape (n_rounds,)
        The weighted error :math:`\mathrm{err}_m` of each estimator on the
        distribution it was trained against.
    classes_ : ndarray of shape (n_classes,)
        Sorted unique labels. ``predict`` returns values from here;
        ``predict_proba`` / ``decision_function`` columns are in this
        order.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_importances_ : ndarray of shape (n_features,)
        The :math:`\alpha`-weighted mean of the per-estimator Gini
        importances, renormalised to sum to 1 (all zeros if no estimator
        ever splits).

    Notes
    -----
    SAMME minimises a multi-class exponential loss by forward stagewise
    additive modelling; the weighted-error fit, the estimator weight
    :math:`\alpha_m = \log\frac{1-\mathrm{err}_m}{\mathrm{err}_m}
    + \log(K-1)`, and the multiplicative weight update all fall out of that
    optimisation. Training uses no randomness (the default stump is
    deterministic), so a fit is reproducible without a ``random_state``.
    Boosting rounds are sequential by nature — there is no ``n_jobs``.
    ``AdaBoostRegressor`` (AdaBoost.R2) and ``SAMME.R`` are out of scope.
    Full walkthrough: ``docs/derivations/adaboost.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.ensemble import AdaBoostClassifier
    >>> X = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> model = AdaBoostClassifier(n_estimators=10).fit(X, y)
    >>> model.predict(np.array([[3.0], [9.0]])).tolist()
    [0.0, 1.0]
    >>> len(model.estimators_)  # stopped early: one stump already separates them
    1

    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def fit(
        self,
        X: FeatureMatrix,
        y: TargetVector,
        sample_weight: FloatArray | None = None,
    ) -> AdaBoostClassifier:
        """Run the SAMME boosting loop on ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training labels — any number of classes.
        sample_weight : ndarray of shape (n_samples,), optional
            Initial per-row weights (non-negative, not all zero). ``None``
            starts every row at ``1 / n_samples``; a passed vector is
            normalised to sum to 1.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``n_estimators < 1``, ``learning_rate <= 0``,
            ``max_depth < 1``, ``sample_weight`` is malformed, or the very
            first weak learner fails to beat random guessing.

        """
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}.")
        if self.learning_rate <= 0.0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}.")
        if self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {self.max_depth}.")

        X, y = check_X_y(X, y)
        w = check_sample_weight(sample_weight, X.shape[0])
        w = w / w.sum()  # the boosting loop keeps the weights normalised

        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self.n_features_in_ = X.shape[1]
        random_guess_error = 1.0 - 1.0 / self.n_classes_

        self.estimators_ = []
        weights: list[float] = []
        errors: list[float] = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, y, sample_weight=w)
            err = _weighted_error(y, tree.predict(X), w)

            if err <= 0.0:  # perfect on the reweighted data -> nothing left to boost
                self.estimators_.append(tree)
                weights.append(1.0)
                errors.append(0.0)
                break
            if err >= random_guess_error:  # no better than guessing
                if not self.estimators_:
                    raise ValueError(
                        "The base learner could not beat random guessing on the "
                        "first round; AdaBoost cannot be fit on this data."
                    )
                # Defensive guard, mirroring scikit-learn. Unreachable with a
                # decision-tree weak learner after round 1: a stump can always
                # match the constant weighted-majority predictor, whose error
                # is 1 - max_k W_k <= 1 - 1/K, so `err < random_guess_error`
                # strictly unless the class weights are exactly uniform.
                break  # pragma: no cover  -- keep the estimators we have

            alpha = _samme_alpha(err, self.n_classes_, self.learning_rate)
            miss = tree.predict(X) != y
            w = w * np.exp(alpha * miss)
            w = w / w.sum()  # renormalise for the next round

            self.estimators_.append(tree)
            weights.append(alpha)
            errors.append(err)

        self.estimator_weights_ = np.array(weights)
        self.estimator_errors_ = np.array(errors)

        stacked = np.array([tree.feature_importances_ for tree in self.estimators_])
        normalised_weights = self.estimator_weights_ / self.estimator_weights_.sum()
        importances = normalised_weights @ stacked  # alpha-weighted mean
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

    def _decision(self, X: FeatureMatrix) -> FloatArray:
        """Validate ``X`` and return the ``(n_samples, n_classes)`` SAMME score."""
        X = self._check_predict_input(X)
        return _samme_decision(
            self.estimators_, self.estimator_weights_, self.classes_, X
        )

    def decision_function(self, X: FeatureMatrix) -> FloatArray:
        """Return the SAMME score.

        Shape ``(n_samples, n_classes)`` for a multiclass problem;
        ``(n_samples,)`` for a binary one, as the signed margin
        ``score[:, 1] - score[:, 0]`` (positive favours ``classes_[1]``) —
        scikit-learn's convention.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        score = self._decision(X)
        if self.n_classes_ == 2:
            return score[:, 1] - score[:, 0]
        return score

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict a class label for each row of ``X``.

        The alpha-weighted vote: ``classes_[argmax(score)]``, ties
        resolving to the lowest label.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        score = self._decision(X)
        return self.classes_[np.argmax(score, axis=1)]

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return calibrated class scores, shape ``(n_samples, n_classes)``.

        ``softmax`` of the SAMME score divided by ``n_classes - 1`` — a
        monotone transform of the vote margin, matching scikit-learn.
        Rows sum to 1 and columns follow ``classes_``. This is a score
        calibration, **not** a Bayes posterior.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        score = self._decision(X)
        return softmax(score / (self.n_classes_ - 1), axis=1)

    def staged_predict(self, X: FeatureMatrix) -> Iterator[TargetVector]:
        """Yield the prediction of the ensemble after each boosting round.

        The ``t``-th yielded array is ``predict`` restricted to the first
        ``t`` estimators — the sequence that traces the boosting curve.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        X = self._check_predict_input(X)
        for t in range(1, len(self.estimators_) + 1):
            score = _samme_decision(
                self.estimators_[:t], self.estimator_weights_[:t], self.classes_, X
            )
            yield self.classes_[np.argmax(score, axis=1)]

    def staged_score(self, X: FeatureMatrix, y: TargetVector) -> Iterator[float]:
        """Yield the accuracy of the ensemble after each boosting round."""
        for prediction in self.staged_predict(X):
            yield accuracy_score(y, prediction)

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``."""
        return accuracy_score(y, self.predict(X))
