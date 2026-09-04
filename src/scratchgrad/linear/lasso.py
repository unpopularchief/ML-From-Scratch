r"""Lasso regression — L1-penalised least squares via coordinate descent.

Fits :math:`\hat{y} = Xw + b` by minimising

.. math::
    J(w, b) = \frac{1}{2n}\lVert y - Xw - b\mathbf{1}\rVert_2^2
              + \alpha\lVert w\rVert_1

The L1 term is not differentiable where any :math:`w_j = 0`, so there is no
normal equation. Instead the objective is minimised one coordinate at a
time: with all other weights fixed, the optimal :math:`w_j` is given in
closed form by the soft-thresholding operator

.. math::
    w_j = \frac{S(\rho_j,\ \alpha)}{z_j}, \qquad
    S(\rho, \alpha) = \operatorname{sign}(\rho)\,(\lvert\rho\rvert - \alpha)_+

with :math:`\rho_j = \frac1n x_j^\top r_j` (``r_j`` the partial residual
with feature ``j`` added back) and :math:`z_j = \frac1n\lVert x_j\rVert_2^2`.
Cyclic sweeps over the coordinates converge to the global optimum because
the penalty is separable and the objective is convex.

The ``alpha`` here matches ``sklearn.linear_model.Lasso(alpha=...)`` — note
the :math:`\tfrac{1}{2n}` on the data term, sklearn's Lasso convention (a
different scaling than this project's :class:`~scratchgrad.linear.Ridge`,
which follows sklearn's *Ridge* convention). ``alpha=0`` reduces to
:class:`~scratchgrad.linear.LinearRegression`.

Full derivation: ``docs/derivations/lasso.md``.
"""

from __future__ import annotations

import warnings

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.exceptions import ConvergenceWarning
from scratchgrad.metrics.regression import r2_score
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y


def _soft_threshold(rho: float, alpha: float) -> float:
    r"""Soft-thresholding operator :math:`S(\rho, \alpha)`.

    :math:`S(\rho, \alpha) = \operatorname{sign}(\rho)\,
    (\lvert\rho\rvert - \alpha)_+` — shrink ``rho`` toward zero by
    ``alpha``, snapping to exactly zero on :math:`[-\alpha, \alpha]`. This
    is the 1-D minimiser of :math:`\frac12(w - \rho)^2 + \alpha\lvert
    w\rvert` and the reason Lasso produces exact zeros.
    """
    if rho > alpha:
        return rho - alpha
    if rho < -alpha:
        return rho + alpha
    return 0.0


def _lasso_objective(
    X: FloatArray,
    y: TargetVector,
    w: FloatArray,
    b: float,
    alpha: float,
) -> float:
    r"""Lasso objective.

    :math:`J = \frac{1}{2n}\lVert y - Xw - b\rVert_2^2 + \alpha\lVert
    w\rVert_1`.
    """
    n = X.shape[0]
    residual = y - X @ w - b  # r = y − Xw − b
    return float(residual @ residual / (2 * n) + alpha * np.sum(np.abs(w)))


class Lasso(Estimator):
    r"""Lasso regression: least squares with an L1 penalty on the weights.

    Solved by cyclic coordinate descent. The L1 penalty drives some
    coefficients to *exactly* zero, so Lasso doubles as variable
    selection — unlike :class:`~scratchgrad.linear.Ridge`, whose L2
    penalty only shrinks.

    Parameters
    ----------
    alpha : float, default=1.0
        L1 regularisation strength. ``0`` recovers ordinary least squares
        (:class:`~scratchgrad.linear.LinearRegression`); larger values
        zero out more coefficients. Matches
        ``sklearn.linear_model.Lasso``'s ``alpha`` (objective scaled by
        ``1 / (2 * n_samples)``).
    fit_intercept : bool, default=True
        If ``True``, ``X`` and ``y`` are centered before the fit and the
        intercept is recovered as ``y_mean - x_mean @ coef_``; the
        intercept is never penalised. If ``False``, the model passes
        through the origin (``intercept_`` is fixed at ``0.0``).
    max_iter : int, default=1000
        Maximum number of full coordinate-descent sweeps.
    tol : float, default=1e-4
        Stop once the largest coefficient change in a sweep falls below
        this.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted weight vector :math:`w`, typically sparse.
    intercept_ : float
        Fitted intercept :math:`b` (``0.0`` when ``fit_intercept=False``).
    n_iter_ : int
        Number of coordinate-descent sweeps run.

    Notes
    -----
    The objective minimised is
    :math:`J(w, b) = \frac{1}{2n}\lVert y - Xw - b\rVert_2^2
    + \alpha\lVert w\rVert_1`. Each coordinate update solves its 1-D
    subproblem exactly via :func:`_soft_threshold`; the full residual is
    maintained across updates so a sweep costs :math:`O(nd)`. See
    ``docs/derivations/lasso.md``.

    ``alpha`` penalises every weight equally, so on raw, differently-scaled
    features it bites unevenly. This estimator never rescales its inputs —
    standardise with
    :class:`scratchgrad.preprocessing.StandardScaler` first if you want
    ``alpha`` to mean the same thing across features.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.linear import Lasso
    >>> X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
    >>> y = np.array([1.0, 3.0, 5.0, 7.0])  # y = 2*x0 + 1; x1 is a constant column
    >>> model = Lasso(alpha=0.1).fit(X, y)
    >>> float(model.coef_[1])  # a constant feature carries no information
    0.0

    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: FeatureMatrix, y: TargetVector) -> Lasso:
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
            If ``alpha`` is negative or ``max_iter`` is less than 1.

        """
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        X, y = check_X_y(X, y)

        X_c, y_c, x_mean, y_mean = self._center(X, y)
        w = self._coordinate_descent(X_c, y_c)

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w) if self.fit_intercept else 0.0
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

    def _center(
        self, X: FeatureMatrix, y: TargetVector
    ) -> tuple[FloatArray, TargetVector, FloatArray, float]:
        r"""Center ``X`` and ``y`` when ``fit_intercept`` is set.

        Coordinate descent then runs on a pure no-intercept problem and
        the intercept is recovered afterwards as
        :math:`b = \bar y - \bar x^\top w` (the exact minimiser over the
        unpenalised intercept for any ``w``).
        """
        if not self.fit_intercept:
            zeros = np.zeros(X.shape[1])
            return X, y, zeros, 0.0
        x_mean = X.mean(axis=0)
        y_mean = float(y.mean())
        return X - x_mean, y - y_mean, x_mean, y_mean

    def _coordinate_descent(self, X: FloatArray, y: TargetVector) -> FloatArray:
        r"""Minimise :math:`\frac1{2n}\lVert y - Xw\rVert_2^2 + \alpha\lVert w\rVert_1`.

        Cyclic sweeps over the coordinates. For coordinate ``j`` the
        update is :math:`w_j \leftarrow S(\rho_j, \alpha) / z_j` with
        :math:`\rho_j = \frac1n x_j^\top r + z_j w_j` and
        :math:`z_j = \frac1n\lVert x_j\rVert_2^2`; the residual
        :math:`r = y - Xw` is kept current so each sweep costs
        :math:`O(nd)`.
        """
        n, d = X.shape
        w = np.zeros(d)
        z = np.sum(X**2, axis=0) / n  # zⱼ = ‖xⱼ‖² / n, constant across sweeps
        residual = y - X @ w  # r = y − Xw  (== y, since w starts at 0)

        self.n_iter_ = 0
        for sweep in range(1, self.max_iter + 1):
            max_change = 0.0
            for j in range(d):
                if z[j] == 0.0:
                    continue  # constant feature carries no information
                # ρⱼ = (1/n) xⱼᵀ r_j, with r_j = r + wⱼ xⱼ the partial residual
                rho_j = X[:, j] @ residual / n + w[j] * z[j]
                w_j_new = _soft_threshold(rho_j, self.alpha) / z[j]  # S(ρⱼ, α) / zⱼ
                change = w_j_new - w[j]
                if change != 0.0:
                    residual -= change * X[:, j]  # restore r = y − Xw
                    w[j] = w_j_new
                    max_change = max(max_change, abs(change))
            self.n_iter_ = sweep
            if max_change < self.tol:
                break
        else:
            warnings.warn(
                f"Coordinate descent did not converge in max_iter={self.max_iter} "
                f"(max coefficient change = {max_change:.2e}, tol={self.tol:.2e}). "
                f"Try a larger max_iter or standardizing the features.",
                ConvergenceWarning,
                stacklevel=3,
            )
        return w
