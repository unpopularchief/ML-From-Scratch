r"""Linear SVM — soft-margin primal, subgradient descent.

Fits a maximum-margin linear separator by minimising the soft-margin primal
objective directly (not the dual — no kernel trick here, see
``plan.md`` §1):

.. math::
    J(w, b) = \frac12 \lVert w \rVert_2^2 + C \sum_{i=1}^n
              \max\big(0,\; 1 - y_i(w^\top x_i + b)\big), \qquad y_i \in \{-1, +1\}

matching ``sklearn.svm.LinearSVC(loss="hinge")``'s documented primal exactly,
so our ``C`` equals theirs directly. The hinge term is not differentiable at
:math:`y_i z_i = 1`, so there is no gradient descent in the ordinary sense —
:func:`_svm_subgradient` picks a consistent subgradient (the flat side of the
kink), and :class:`LinearSVM` runs the **subgradient method** with a
diminishing step size :math:`\eta_t = \mathrm{lr}/\sqrt t` and best-iterate
("pocket") tracking, both needed because a fixed step size on a non-smooth
objective need not converge and the objective is not monotone step-to-step
the way smooth gradient descent's is.

Deterministic: :math:`w^{(0)} = b^{(0)} = 0` and every step is a fixed
function of the data, so there is no ``random_state`` parameter.

Full derivation: ``docs/derivations/linear_svm.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y


def _svm_objective(
    X: FloatArray, y: FloatArray, w: FloatArray, b: float, C: float
) -> float:
    r"""Soft-margin primal objective.

    :math:`J(w, b) = \frac12\lVert w\rVert_2^2 + C\sum_i \max(0,
    1 - y_i(w^\top x_i + b))`, with ``y`` encoded :math:`\{-1, +1\}`.
    """
    margin = y * (X @ w + b)  # m = y ⊙ (Xw + b)
    hinge = np.maximum(0.0, 1.0 - margin)  # max(0, 1 − m)
    return float(0.5 * (w @ w) + C * np.sum(hinge))


def _svm_subgradient(
    X: FloatArray, y: FloatArray, w: FloatArray, b: float, C: float
) -> tuple[FloatArray, float]:
    r"""Compute a subgradient ``(g_w, g_b)`` of :func:`_svm_objective` at ``(w, b)``.

    :math:`g_w = w - C\sum_{i \in \mathcal V} y_i x_i`, :math:`g_b =
    -C\sum_{i \in \mathcal V} y_i`, where :math:`\mathcal V = \{i : y_i(w^\top
    x_i + b) < 1\}` is the set of margin-violating samples. Picks the
    flat-side subgradient (``0``) exactly on the margin, matching the
    Lasso-style "consistent choice at the kink" convention.
    """
    margin = y * (X @ w + b)  # m = y ⊙ (Xw + b)
    violated = margin < 1.0  # V = {i : m_i < 1}
    # X[violated] on an empty mask is an empty (0, d) slice, so this already
    # evaluates to zeros(d) with no violations — no special case needed.
    g_w = w - C * (X[violated].T @ y[violated])  # w − C Σ_{i∈V} yᵢxᵢ
    g_b = -C * float(np.sum(y[violated]))  # −C Σ_{i∈V} yᵢ
    return g_w, g_b


class LinearSVM(Estimator):
    r"""Soft-margin linear support vector classifier (primal, subgradient descent).

    Parameters
    ----------
    C : float, default=1.0
        Regularisation strength: how heavily margin violations are
        penalised relative to margin width. Matches
        ``sklearn.svm.LinearSVC``'s ``C`` exactly (same primal objective,
        no rescaling). Larger ``C`` fits the training data more tightly
        (a "harder" margin); smaller ``C`` favors a wider margin at the
        cost of tolerating more violations.
    fit_intercept : bool, default=True
        If ``True``, learn an intercept ``b`` alongside ``w``. If
        ``False``, the decision boundary passes through the origin
        (``intercept_`` fixed at ``0.0``).
    lr : float, default=0.01
        Base subgradient-descent step size. The step actually taken at
        iteration ``t`` is ``lr / sqrt(t)`` (diminishing schedule — see
        ``docs/derivations/linear_svm.md`` §4).
    max_iter : int, default=1000
        Number of subgradient steps to run. There is no ``tol``/early
        stopping here (§5 of the derivation) — this always runs exactly
        ``max_iter`` steps and keeps the lowest-objective iterate seen.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Weight vector :math:`w` at the best (lowest-``J``) iterate seen.
    intercept_ : float
        Intercept :math:`b` at that same iterate (``0.0`` when
        ``fit_intercept=False``).
    classes_ : ndarray of shape (2,)
        The two class labels seen in ``y``, sorted. ``predict`` returns
        values drawn from here; the larger label is mapped internally to
        :math:`+1`.
    n_iter_ : int
        Always equal to ``max_iter`` — the number of subgradient steps
        run (not a convergence claim; see the derivation §5).

    Notes
    -----
    Minimises :math:`J(w, b) = \frac12\lVert w\rVert_2^2 + C\sum_i
    \max(0, 1 - y_i(w^\top x_i + b))` by the subgradient method: at each
    step, :math:`(w, b) \leftarrow (w, b) - \eta_t\,(g_w, g_b)` with
    :math:`\eta_t = \mathrm{lr}/\sqrt{t}` and :math:`(g_w, g_b)` a
    subgradient of :math:`J` (:func:`_svm_subgradient`). No
    ``random_state`` — the whole algorithm is deterministic. See
    ``docs/derivations/linear_svm.md``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.svm import LinearSVM
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = LinearSVM().fit(X, y)
    >>> model.predict(np.array([[-1.5], [1.5]])).tolist()
    [0.0, 1.0]
    >>> bool(model.coef_[0] > 0)  # larger x -> more positive decision score
    True

    """

    def __init__(
        self,
        C: float = 1.0,
        fit_intercept: bool = True,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.C = C
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, X: FeatureMatrix, y: TargetVector) -> LinearSVM:
        """Fit ``coef_`` and ``intercept_`` to ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Binary training labels — exactly two distinct numeric values.
            The larger is mapped internally to :math:`+1`.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``C`` is not positive, ``lr`` is not positive, ``max_iter``
            is less than 1, or ``y`` does not have exactly two distinct
            values.

        """
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}.")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")

        X, y = check_X_y(X, y)
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"LinearSVM supports only binary classification, got "
                f"{classes.shape[0]} distinct values in y."
            )
        self.classes_ = classes
        target = np.where(y == classes[1], 1.0, -1.0)  # labels -> {-1, +1}

        self.coef_, self.intercept_ = self._fit_subgradient_descent(X, target)
        self.n_iter_ = self.max_iter
        return self

    def decision_function(self, X: FeatureMatrix) -> TargetVector:
        """Return the signed decision score ``X @ coef_ + intercept_``.

        Positive values predict ``classes_[1]``, negative predict
        ``classes_[0]``; the magnitude is the (unnormalised) confidence.
        """
        check_is_fitted(self, "coef_")
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.coef_.shape[0]}."
            )
        return X @ self.coef_ + self.intercept_  # z = Xw + b

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict class labels for ``X`` by the sign of the decision score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict for; ``n_features`` must match training.

        Returns
        -------
        ndarray of shape (n_samples,)
            Each entry is one of the two values in ``classes_``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        positive = self.decision_function(X) >= 0.0
        return self.classes_[positive.astype(int)]

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.accuracy_score`.
        """
        return accuracy_score(y, self.predict(X))

    def _fit_subgradient_descent(
        self, X: FloatArray, target: FloatArray
    ) -> tuple[FloatArray, float]:
        r"""Run the subgradient method, returning the best-``J`` ``(w, b)``.

        :math:`\eta_t = \mathrm{lr}/\sqrt t` diminishing step size;
        ``(w_best, b_best)`` tracks the lowest objective seen across all
        ``max_iter`` steps, since ``J`` is not guaranteed to decrease at
        every individual step the way a smooth gradient descent's would
        (``docs/derivations/linear_svm.md`` §4).
        """
        d = X.shape[1]
        w = np.zeros(d)
        b = 0.0
        w_best, b_best = w.copy(), b
        best_objective = _svm_objective(X, target, w, b, self.C)

        for step in range(1, self.max_iter + 1):
            g_w, g_b = _svm_subgradient(X, target, w, b, self.C)
            eta = self.lr / np.sqrt(step)  # η_t = lr / √t
            w = w - eta * g_w
            if self.fit_intercept:
                b = b - eta * g_b
            objective = _svm_objective(X, target, w, b, self.C)
            if objective < best_objective:
                best_objective = objective
                w_best, b_best = w.copy(), b

        return w_best, (b_best if self.fit_intercept else 0.0)
