r"""Ordinary least squares linear regression.

Fits :math:`\hat{y} = Xw + b` by minimizing the mean squared error

.. math::
    J(\theta) = \frac{1}{n} \lVert \tilde{X}\theta - y \rVert_2^2

where :math:`\tilde{X} = [\mathbf{1} \; X]` folds the intercept into the
weight vector :math:`\theta = [b, w]`. Two solvers are offered:

- ``"normal"`` — solve the normal equation
  :math:`\tilde{X}^\top\tilde{X}\,\theta = \tilde{X}^\top y` directly (via
  ``np.linalg.lstsq``, i.e. an SVD least-squares solve rather than an
  explicit matrix inverse).
- ``"gd"`` — batch gradient descent on :math:`J`, using the gradient
  :math:`\nabla_\theta J = \frac{2}{n}\tilde{X}^\top(\tilde{X}\theta - y)`.

Full derivation: ``docs/derivations/linear_regression.md``.
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


def _mse_objective(X_aug: FloatArray, y: TargetVector, theta: FloatArray) -> float:
    r"""Least-squares objective.

    :math:`J(\theta) = \frac{1}{n}\lVert \tilde X\theta - y\rVert_2^2`.
    """
    residual = X_aug @ theta - y  # r = X̃θ − y
    return float(np.mean(residual**2))


def _mse_gradient(X_aug: FloatArray, y: TargetVector, theta: FloatArray) -> FloatArray:
    r"""Gradient of :func:`_mse_objective`.

    :math:`\nabla_\theta J = \frac{2}{n}\tilde X^\top(\tilde X\theta - y)`.
    """
    n = X_aug.shape[0]
    residual = X_aug @ theta - y  # r = X̃θ − y
    return (2.0 / n) * (X_aug.T @ residual)  # ∇J = (2/n) X̃ᵀr


class LinearRegression(Estimator):
    r"""Ordinary least squares linear regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        If ``True``, prepend a constant-1 column to ``X`` so an intercept
        term is learned. If ``False``, the model passes through the origin
        (``intercept_`` is fixed at ``0.0``) — use only when the data is
        already centered.
    solver : {"normal", "gd"}, default="normal"
        ``"normal"`` solves the normal equation in one shot (exact, no
        hyperparameters). ``"gd"`` runs batch gradient descent and uses
        ``lr``, ``max_iter`` and ``tol``.
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
    The objective minimized is the mean squared error
    :math:`J(\theta) = \frac1n\lVert \tilde X\theta - y\rVert_2^2`, with
    :math:`\tilde X = [\mathbf 1\; X]` and :math:`\theta = [b, w]`.
    ``solver="normal"`` solves :math:`\tilde X^\top\tilde X\,\theta =
    \tilde X^\top y`; ``solver="gd"`` follows
    :math:`\nabla_\theta J = \frac2n \tilde X^\top(\tilde X\theta - y)`.
    See ``docs/derivations/linear_regression.md``.

    Gradient descent converges much faster on comparably-scaled features.
    This estimator never rescales its inputs — standardize with
    :class:`scratchgrad.preprocessing.StandardScaler` before ``fit`` when
    using ``solver="gd"``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.linear import LinearRegression
    >>> X = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2x + 1
    >>> model = LinearRegression().fit(X, y)
    >>> float(np.round(model.coef_[0], 6)), round(model.intercept_, 6)
    (2.0, 1.0)

    """

    def __init__(
        self,
        fit_intercept: bool = True,
        solver: str = "normal",
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: FeatureMatrix, y: TargetVector) -> LinearRegression:
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
            If ``solver`` is not one of ``{"normal", "gd"}``, or (for
            ``"gd"``) ``max_iter`` is less than 1.

        """
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")
        X, y = check_X_y(X, y)
        X_aug = self._augment(X)

        if self.solver == "normal":
            theta = self._fit_normal_equation(X_aug, y)
        else:
            theta = self._fit_gradient_descent(X_aug, y)

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

    def _fit_normal_equation(self, X_aug: FloatArray, y: TargetVector) -> FloatArray:
        r"""Solve :math:`\tilde X^\top\tilde X\,\theta = \tilde X^\top y` for ``theta``.

        Uses ``np.linalg.lstsq`` (SVD least-squares) rather than forming
        :math:`(\tilde X^\top\tilde X)^{-1}` — same solution when the
        system is full rank, but numerically better behaved and still
        defined (minimum-norm) when it is not.
        """
        theta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return theta

    def _fit_gradient_descent(self, X_aug: FloatArray, y: TargetVector) -> FloatArray:
        r"""Minimize :math:`J(\theta)` by batch gradient descent, starting from zero."""
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")

        theta = np.zeros(X_aug.shape[1])
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            gradient = _mse_gradient(X_aug, y, theta)  # ∇J(θ⁽ᵗ⁾)
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
