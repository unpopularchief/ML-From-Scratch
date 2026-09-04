r"""Ridge regression — L2-penalised least squares.

Fits :math:`\hat{y} = Xw + b` by minimising the mean squared error with an
L2 penalty on the weights (but **not** the intercept):

.. math::
    J(\theta) = \frac{1}{n}\Big( \lVert \tilde{X}\theta - y \rVert_2^2
                + \alpha \lVert w \rVert_2^2 \Big)

where :math:`\tilde{X} = [\mathbf{1} \; X]` folds the intercept into the
parameter vector :math:`\theta = [b, w]` and :math:`D = \operatorname{diag}
(0, 1, \dots, 1)` picks out the penalised (non-intercept) entries. Two
solvers are offered:

- ``"normal"`` — solve the regularised normal equation
  :math:`(\tilde{X}^\top\tilde{X} + \alpha D)\,\theta = \tilde{X}^\top y`
  directly (via ``np.linalg.solve``).
- ``"gd"`` — batch gradient descent on :math:`J`, using the gradient
  :math:`\nabla_\theta J = \frac{2}{n}\big(\tilde{X}^\top(\tilde{X}\theta - y)
  + \alpha D\theta\big)`.

The ``alpha`` here matches ``sklearn.linear_model.Ridge(alpha=...)``: the
:math:`\frac1n` wraps both terms, so scaling the whole objective does not
move its minimiser. ``alpha=0`` reduces exactly to
:class:`~scratchgrad.linear.LinearRegression`.

Full derivation: ``docs/derivations/ridge.md``.
"""

from __future__ import annotations

import warnings

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.exceptions import ConvergenceWarning
from scratchgrad.metrics.regression import r2_score
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y

_SOLVERS = ("normal", "gd")


def _ridge_objective(
    X_aug: FloatArray,
    y: TargetVector,
    theta: FloatArray,
    alpha: float,
    penalty_mask: FloatArray,
) -> float:
    r"""Ridge objective.

    :math:`J(\theta) = \frac{1}{n}\big(\lVert \tilde X\theta - y\rVert_2^2
    + \alpha\lVert w\rVert_2^2\big)`, where ``penalty_mask`` is the diagonal
    of :math:`D` (``0`` in the intercept slot, ``1`` elsewhere).
    """
    n = X_aug.shape[0]
    residual = X_aug @ theta - y  # r = X̃θ − y
    penalty = alpha * np.sum((penalty_mask * theta) ** 2)  # α‖w‖²
    return float((residual @ residual + penalty) / n)


def _ridge_gradient(
    X_aug: FloatArray,
    y: TargetVector,
    theta: FloatArray,
    alpha: float,
    penalty_mask: FloatArray,
) -> FloatArray:
    r"""Gradient of :func:`_ridge_objective`.

    :math:`\nabla_\theta J = \frac{2}{n}\big(\tilde X^\top(\tilde X\theta - y)
    + \alpha D\theta\big)`.
    """
    n = X_aug.shape[0]
    residual = X_aug @ theta - y  # r = X̃θ − y
    data_term = X_aug.T @ residual  # X̃ᵀr
    penalty_term = alpha * (penalty_mask * theta)  # α D θ
    return (2.0 / n) * (data_term + penalty_term)  # ∇J


class Ridge(Estimator):
    r"""Ridge regression: least squares with an L2 penalty on the weights.

    Parameters
    ----------
    alpha : float, default=1.0
        L2 regularisation strength. ``0`` recovers ordinary least squares
        (:class:`~scratchgrad.linear.LinearRegression`); larger values
        shrink ``coef_`` further toward zero (but never exactly to zero —
        that is Lasso). Matches ``sklearn.linear_model.Ridge``'s ``alpha``.
    fit_intercept : bool, default=True
        If ``True``, prepend a constant-1 column to ``X`` so an intercept
        term is learned. The intercept is **not** penalised. If ``False``,
        the model passes through the origin (``intercept_`` is fixed at
        ``0.0``) — use only when the data is already centered.
    solver : {"normal", "gd"}, default="normal"
        ``"normal"`` solves the regularised normal equation in one shot
        (exact). ``"gd"`` runs batch gradient descent and uses ``lr``,
        ``max_iter`` and ``tol``.
    lr : float, default=0.01
        Learning rate for ``solver="gd"``. Ignored otherwise.
    max_iter : int, default=1000
        Maximum gradient-descent steps for ``solver="gd"``. Ignored
        otherwise.
    tol : float, default=1e-6
        Gradient-descent stops once the largest absolute gradient
        component falls below this. Ignored for ``solver="normal"``.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted weight vector :math:`w`.
    intercept_ : float
        Fitted intercept :math:`b` (``0.0`` when ``fit_intercept=False``).
    n_iter_ : int
        Number of gradient-descent steps run. Only set for
        ``solver="gd"``.

    Notes
    -----
    The objective minimised is
    :math:`J(\theta) = \frac1n\big(\lVert \tilde X\theta - y\rVert_2^2
    + \alpha\lVert w\rVert_2^2\big)`, with :math:`\tilde X = [\mathbf 1\; X]`,
    :math:`\theta = [b, w]` and :math:`D = \operatorname{diag}(0, 1, \dots, 1)`.
    ``solver="normal"`` solves :math:`(\tilde X^\top\tilde X + \alpha D)\,\theta
    = \tilde X^\top y`; ``solver="gd"`` follows
    :math:`\nabla_\theta J = \frac2n\big(\tilde X^\top(\tilde X\theta - y)
    + \alpha D\theta\big)`. See ``docs/derivations/ridge.md``.

    Gradient descent converges much faster on comparably-scaled features.
    This estimator never rescales its inputs — standardise with
    :class:`scratchgrad.preprocessing.StandardScaler` before ``fit`` when
    using ``solver="gd"``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.linear import Ridge
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1
    >>> model = Ridge(alpha=0.0).fit(X, y)  # alpha=0 -> ordinary least squares
    >>> float(np.round(model.coef_[0], 6)), round(model.intercept_, 6)
    (2.0, 1.0)

    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        solver: str = "normal",
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: FeatureMatrix, y: TargetVector) -> Ridge:
        """Fit ``coef_`` and ``intercept_`` to ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``alpha`` is negative, ``solver`` is not one of
            ``{"normal", "gd"}``, or (for ``"gd"``) ``max_iter`` is less
            than 1.

        """
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        X, y = check_X_y(X, y)
        X_aug = self._augment(X)
        penalty_mask = self._penalty_mask(X_aug.shape[1])

        if self.solver == "normal":
            theta = self._fit_normal_equation(X_aug, y, penalty_mask)
        else:
            theta = self._fit_gradient_descent(X_aug, y, penalty_mask)

        if self.fit_intercept:
            self.intercept_ = float(theta[0])  # b is the leading augmented column
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        return self

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict targets for ``X`` as ``X @ coef_ + intercept_``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict for; ``n_features`` must match training.

        Returns
        -------
        ndarray of shape (n_samples,)

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than the training
            data.

        """
        check_is_fitted(self, "coef_")
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.coef_.shape[0]}."
            )
        return X @ self.coef_ + self.intercept_  # ŷ = Xw + b

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the :math:`R^2` of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.r2_score`. 1.0 is a perfect fit,
        0.0 matches a constant "predict the mean" baseline, negative is
        worse than that baseline.
        """
        return r2_score(y, self.predict(X))

    def _augment(self, X: FeatureMatrix) -> FloatArray:
        """Prepend a constant-1 column when ``fit_intercept`` is set."""
        if not self.fit_intercept:
            return X
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])  # X̃ = [1 | X]

    def _penalty_mask(self, n_params: int) -> FloatArray:
        r"""Diagonal of :math:`D`: ``0`` for the intercept, ``1`` per weight.

        Multiplying ``theta`` by this is :math:`D\theta`; adding
        ``alpha`` times it to the Gram diagonal is :math:`+\,\alpha D`.
        """
        mask = np.ones(n_params)
        if self.fit_intercept:
            mask[0] = 0.0  # the intercept term is never penalised
        return mask

    def _fit_normal_equation(
        self, X_aug: FloatArray, y: TargetVector, penalty_mask: FloatArray
    ) -> FloatArray:
        r"""Solve :math:`(\tilde X^\top\tilde X + \alpha D)\theta = \tilde X^\top y`.

        Uses ``np.linalg.solve`` rather than forming an explicit inverse.
        For ``alpha > 0`` the left-hand matrix is symmetric positive
        definite (the :math:`\alpha D` term lifts every non-intercept
        eigenvalue by ``alpha``), so the solve is well-posed even when
        :math:`\tilde X` is rank-deficient. ``alpha=0`` on collinear data
        is the degenerate case — use ``LinearRegression`` there instead.
        """
        gram = X_aug.T @ X_aug + self.alpha * np.diag(penalty_mask)  # X̃ᵀX̃ + αD
        target = X_aug.T @ y  # X̃ᵀy
        return np.linalg.solve(gram, target)

    def _fit_gradient_descent(
        self, X_aug: FloatArray, y: TargetVector, penalty_mask: FloatArray
    ) -> FloatArray:
        r"""Minimise :math:`J(\theta)` by batch gradient descent, starting from zero."""
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")

        theta = np.zeros(X_aug.shape[1])
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            gradient = _ridge_gradient(
                X_aug, y, theta, self.alpha, penalty_mask
            )  # ∇J(θ⁽ᵗ⁾)
            theta = theta - self.lr * gradient  # θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − lr·∇J
            self.n_iter_ = iteration
            if np.max(np.abs(gradient)) < self.tol:
                break
        else:
            max_grad = float(np.max(np.abs(gradient)))
            warnings.warn(
                f"Gradient descent did not converge in max_iter={self.max_iter} "
                f"(max |gradient| = {max_grad:.2e}, tol={self.tol:.2e}). Try a "
                f"larger max_iter, a different lr, or standardizing the features.",
                ConvergenceWarning,
                stacklevel=3,
            )
        return theta
