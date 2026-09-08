r"""Binary logistic regression — maximum likelihood / cross-entropy.

Models :math:`P(y = 1 \mid x) = \sigma(w^\top x + b)` and fits :math:`\theta =
[b, w]` by minimising the mean negative log-likelihood with an optional L2
penalty on the weights (never the intercept):

.. math::
    J(\theta) = \frac{1}{n}\left[\,\sum_i \big(\mathrm{softplus}(z_i)
                - y_i z_i\big) + \frac{1}{2C}\lVert w \rVert_2^2 \,\right],
    \qquad z = \tilde{X}\theta

where :math:`\tilde{X} = [\mathbf{1} \; X]` folds the intercept in and
:math:`\mathrm{softplus}(z) - y z = -[y\log\sigma(z) + (1 - y)
\log(1 - \sigma(z))]` is the numerically stable form of the per-sample loss.
:math:`J` is convex. Two iterative solvers are offered (there is no closed
form):

- ``"newton"`` — Newton--Raphson, i.e. iteratively reweighted least squares:
  :math:`\theta \leftarrow \theta - H^{-1}g` with
  :math:`H = \frac1n(\tilde{X}^\top S\tilde{X} + \frac1C D)` and
  :math:`S = \mathrm{diag}(p_i(1 - p_i))`. Quadratic convergence, no
  learning rate. The default.
- ``"gd"`` — batch gradient descent on :math:`J`, using
  :math:`\nabla_\theta J = \frac1n(\tilde{X}^\top(p - y) + \frac1C D\theta)`.

The ``C`` here matches ``sklearn.linear_model.LogisticRegression(C=...)``: our
objective is scikit-learn's times :math:`\frac{1}{nC}`, which does not move the
minimiser. ``penalty=None`` drops the L2 term.

Full derivation: ``docs/derivations/logistic_regression.md``.
"""

from __future__ import annotations

import warnings

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.exceptions import ConvergenceWarning
from scratchgrad.metrics.classification import accuracy_score
from scratchgrad.typing import FeatureMatrix, FloatArray, TargetVector
from scratchgrad.utils.math import sigmoid
from scratchgrad.utils.validation import check_array, check_is_fitted, check_X_y

_SOLVERS = ("newton", "gd")
_PENALTIES = ("l2", None)


def _logistic_objective(
    X_aug: FloatArray,
    y: TargetVector,
    theta: FloatArray,
    inv_C: float,
    penalty_mask: FloatArray,
) -> float:
    r"""Mean negative log-likelihood plus the L2 penalty.

    :math:`J(\theta) = \frac1n\big[\sum_i (\mathrm{softplus}(z_i)
    - y_i z_i) + \frac{1}{2C}\lVert w\rVert_2^2\big]`, with
    :math:`z = \tilde{X}\theta`, ``inv_C`` standing in for :math:`1/C`
    (``0`` when there is no penalty) and ``penalty_mask`` the diagonal of
    :math:`D` (``0`` in the intercept slot, ``1`` elsewhere).
    """
    n = X_aug.shape[0]
    z = X_aug @ theta  # z = X̃θ
    # softplus(z) − y·z  is  −[y log σ(z) + (1−y) log(1−σ(z))], computed
    # via logaddexp(0, z) = log(1 + eᶻ) so no exp overflows and no log of a
    # rounded-to-0/1 probability is taken.
    nll = float(np.sum(np.logaddexp(0.0, z) - y * z))  # Σ softplus(zᵢ) − yᵢzᵢ
    penalty = 0.5 * inv_C * float(np.sum((penalty_mask * theta) ** 2))  # ‖w‖²/(2C)
    return (nll + penalty) / n


def _logistic_gradient(
    X_aug: FloatArray,
    y: TargetVector,
    theta: FloatArray,
    inv_C: float,
    penalty_mask: FloatArray,
) -> FloatArray:
    r"""Gradient of :func:`_logistic_objective`.

    :math:`\nabla_\theta J = \frac1n\big(\tilde{X}^\top(p - y)
    + \frac1C D\theta\big)`, with :math:`p = \sigma(\tilde{X}\theta)`.
    """
    n = X_aug.shape[0]
    p = sigmoid(X_aug @ theta)  # p = σ(X̃θ)
    data_term = X_aug.T @ (p - y)  # X̃ᵀ(p − y)
    penalty_term = inv_C * (penalty_mask * theta)  # (1/C) D θ
    return (data_term + penalty_term) / n  # ∇J


def _logistic_hessian(
    X_aug: FloatArray,
    theta: FloatArray,
    inv_C: float,
    penalty_mask: FloatArray,
) -> FloatArray:
    r"""Hessian of :func:`_logistic_objective`.

    :math:`\nabla^2_\theta J = \frac1n\big(\tilde{X}^\top S\tilde{X}
    + \frac1C D\big)`, with :math:`S = \mathrm{diag}(p_i(1 - p_i))`.
    Does not depend on ``y``.
    """
    n = X_aug.shape[0]
    p = sigmoid(X_aug @ theta)  # p = σ(X̃θ)
    s = p * (1.0 - p)  # diagonal of S
    data_term = X_aug.T @ (s[:, None] * X_aug)  # X̃ᵀ S X̃
    penalty_term = inv_C * np.diag(penalty_mask)  # (1/C) D
    return (data_term + penalty_term) / n  # ∇²J


class LogisticRegression(Estimator):
    r"""Binary logistic regression fitted by maximum likelihood.

    Parameters
    ----------
    C : float, default=1.0
        Inverse L2 regularisation strength (smaller ``C`` = stronger
        penalty). Matches ``sklearn.linear_model.LogisticRegression``'s
        ``C``. Ignored when ``penalty=None``.
    penalty : {"l2", None}, default="l2"
        ``"l2"`` adds :math:`\frac{1}{2C}\lVert w\rVert_2^2` to the
        objective (the intercept is **not** penalised). ``None`` fits the
        unregularised MLE — which does not exist for linearly separable
        classes, so both solvers then diverge.
    fit_intercept : bool, default=True
        If ``True``, prepend a constant-1 column to ``X`` so an intercept
        is learned. If ``False``, the decision boundary passes through the
        origin (``intercept_`` fixed at ``0.0``).
    solver : {"newton", "gd"}, default="newton"
        ``"newton"`` runs Newton--Raphson / IRLS (quadratic convergence,
        no ``lr``). ``"gd"`` runs batch gradient descent and uses ``lr``.
        Both use ``max_iter`` and ``tol``.
    lr : float, default=0.1
        Learning rate for ``solver="gd"``. Ignored for ``"newton"``.
    max_iter : int, default=1000
        Maximum solver iterations.
    tol : float, default=1e-6
        Stopping tolerance. For ``"gd"``: the largest absolute gradient
        component. For ``"newton"``: half the Newton decrement
        :math:`\frac12 g^\top H^{-1} g`.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted weight vector :math:`w`.
    intercept_ : float
        Fitted intercept :math:`b` (``0.0`` when ``fit_intercept=False``).
    classes_ : ndarray of shape (2,)
        The two class labels seen in ``y``, sorted. ``predict`` returns
        values drawn from here; the larger label is the "positive" class
        (the one ``predict_proba``'s second column is the probability of).
    n_iter_ : int
        Number of solver iterations run.

    Notes
    -----
    The objective minimised is
    :math:`J(\theta) = \frac1n[\sum_i (\mathrm{softplus}(z_i)
    - y_i z_i) + \frac{1}{2C}\lVert w\rVert_2^2]` with
    :math:`z = \tilde{X}\theta`, :math:`\tilde{X} = [\mathbf 1\; X]`,
    :math:`\theta = [b, w]`. It is convex.
    :math:`\nabla_\theta J = \frac1n(\tilde{X}^\top(p - y) + \frac1C D\theta)`
    and :math:`\nabla^2_\theta J = \frac1n(\tilde{X}^\top S\tilde{X}
    + \frac1C D)` with :math:`p = \sigma(\tilde{X}\theta)` and
    :math:`S = \mathrm{diag}(p_i(1 - p_i))`. See
    ``docs/derivations/logistic_regression.md``.

    Gradient descent converges much faster on comparably-scaled features.
    This estimator never rescales its inputs — standardise with
    :class:`scratchgrad.preprocessing.StandardScaler` before ``fit`` when
    using ``solver="gd"``.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.linear import LogisticRegression
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([0, 0, 1, 1])
    >>> model = LogisticRegression().fit(X, y)
    >>> model.predict(np.array([[-1.5], [1.5]])).tolist()
    [0.0, 1.0]
    >>> bool(model.coef_[0] > 0)  # larger x -> larger P(y=1)
    True

    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str | None = "l2",
        fit_intercept: bool = True,
        solver: str = "newton",
        lr: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.C = C
        self.penalty = penalty
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: FeatureMatrix, y: TargetVector) -> LogisticRegression:
        """Fit ``coef_`` and ``intercept_`` to ``X``, ``y``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training design matrix.
        y : ndarray of shape (n_samples,)
            Binary training labels — exactly two distinct numeric values
            (e.g. ``{0, 1}`` or ``{-1, 1}``). The larger is the positive
            class.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``solver`` is not one of ``{"newton", "gd"}``, ``penalty``
            is not one of ``{"l2", None}``, ``C`` is not positive,
            ``max_iter`` is less than 1, or ``y`` does not have exactly two
            distinct values.

        """
        if self.solver not in _SOLVERS:
            raise ValueError(f"solver must be one of {_SOLVERS}, got {self.solver!r}.")
        if self.penalty not in _PENALTIES:
            raise ValueError(
                f"penalty must be one of {_PENALTIES}, got {self.penalty!r}."
            )
        if self.C <= 0:
            raise ValueError(f"C must be > 0, got {self.C}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")

        X, y = check_X_y(X, y)
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"LogisticRegression supports only binary classification, got "
                f"{classes.shape[0]} distinct values in y."
            )
        self.classes_ = classes
        target = (y == classes[1]).astype(np.float64)  # labels -> {0, 1}

        X_aug = self._augment(X)
        penalty_mask = self._penalty_mask(X_aug.shape[1])
        inv_C = 1.0 / self.C if self.penalty == "l2" else 0.0

        if self.solver == "newton":
            theta = self._fit_newton(X_aug, target, inv_C, penalty_mask)
        else:
            theta = self._fit_gradient_descent(X_aug, target, inv_C, penalty_mask)

        if self.fit_intercept:
            self.intercept_ = float(theta[0])  # b is the leading augmented column
            self.coef_ = theta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = theta
        return self

    def decision_function(self, X: FeatureMatrix) -> TargetVector:
        """Return the logits (log-odds) ``X @ coef_ + intercept_``.

        Positive values predict the positive class, negative the negative
        class; the magnitude is the model's confidence in log-odds units.
        """
        check_is_fitted(self, "coef_")
        X = check_array(X)
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.coef_.shape[0]}."
            )
        return X @ self.coef_ + self.intercept_  # z = Xw + b

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        """Return class probabilities, shape ``(n_samples, 2)``.

        Column 0 is the probability of ``classes_[0]``; column 1 is the
        probability of ``classes_[1]``, i.e. ``sigmoid(X @ coef_ +
        intercept_)``. Each row sums to 1.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        p = sigmoid(self.decision_function(X))  # P(y = classes_[1])
        return np.column_stack([1.0 - p, p])

    def predict(self, X: FeatureMatrix) -> TargetVector:
        """Predict class labels for ``X`` by thresholding probability at 0.5.

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
        positive = self.predict_proba(X)[:, 1] >= 0.5  # σ(X̃θ) >= 0.5  <=>  X̃θ >= 0
        return self.classes_[positive.astype(int)]

    def score(self, X: FeatureMatrix, y: TargetVector) -> float:
        """Return the accuracy of :meth:`predict` against ``y``.

        See :func:`scratchgrad.metrics.accuracy_score`: the fraction of
        samples whose predicted label matches the true label.
        """
        return accuracy_score(y, self.predict(X))

    def _augment(self, X: FeatureMatrix) -> FloatArray:
        """Prepend a constant-1 column when ``fit_intercept`` is set."""
        if not self.fit_intercept:
            return X
        ones = np.ones(X.shape[0])
        return np.column_stack([ones, X])  # X̃ = [1 | X]

    def _penalty_mask(self, n_params: int) -> FloatArray:
        r"""Diagonal of :math:`D`: ``0`` for the intercept, ``1`` per weight.

        Multiplying ``theta`` by this is :math:`D\theta`; it keeps the
        intercept term out of both the penalty and its derivatives.
        """
        mask = np.ones(n_params)
        if self.fit_intercept:
            mask[0] = 0.0  # the intercept term is never penalised
        return mask

    def _fit_newton(
        self,
        X_aug: FloatArray,
        y: TargetVector,
        inv_C: float,
        penalty_mask: FloatArray,
    ) -> FloatArray:
        r"""Minimise :math:`J` by Newton--Raphson (IRLS), starting from zero.

        Each step solves :math:`H\,\delta = g` (via ``np.linalg.solve``,
        not an explicit inverse) and takes :math:`\theta \leftarrow \theta
        - \delta`. Stops when the Newton decrement
        :math:`\frac12 g^\top\delta` drops below ``tol``. With
        ``penalty="l2"`` the Hessian is positive definite, so the solve is
        well-posed; with ``penalty=None`` on separable data it can become
        singular (a documented degenerate case).
        """
        theta = np.zeros(X_aug.shape[1])
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            gradient = _logistic_gradient(X_aug, y, theta, inv_C, penalty_mask)  # g
            hessian = _logistic_hessian(X_aug, theta, inv_C, penalty_mask)  # H
            step = np.linalg.solve(hessian, gradient)  # δ = H⁻¹g
            theta = theta - step  # θ ← θ − δ
            self.n_iter_ = iteration
            decrement = 0.5 * float(gradient @ step)  # ½ gᵀH⁻¹g
            if decrement < self.tol:
                break
        else:
            self._warn_not_converged("Newton", f"Newton decrement = {decrement:.2e}")
        return theta

    def _fit_gradient_descent(
        self,
        X_aug: FloatArray,
        y: TargetVector,
        inv_C: float,
        penalty_mask: FloatArray,
    ) -> FloatArray:
        r"""Minimise :math:`J` by batch gradient descent, starting from zero."""
        theta = np.zeros(X_aug.shape[1])
        self.n_iter_ = 0
        for iteration in range(1, self.max_iter + 1):
            gradient = _logistic_gradient(X_aug, y, theta, inv_C, penalty_mask)  # ∇J
            theta = theta - self.lr * gradient  # θ⁽ᵗ⁺¹⁾ = θ⁽ᵗ⁾ − lr·∇J
            self.n_iter_ = iteration
            if np.max(np.abs(gradient)) < self.tol:
                break
        else:
            max_grad = float(np.max(np.abs(gradient)))
            self._warn_not_converged(
                "Gradient descent", f"max |gradient| = {max_grad:.2e}"
            )
        return theta

    def _warn_not_converged(self, method: str, detail: str) -> None:
        """Emit a uniform :class:`ConvergenceWarning` for either solver."""
        warnings.warn(
            f"{method} did not converge in max_iter={self.max_iter} ({detail}, "
            f"tol={self.tol:.2e}). Try a larger max_iter, a different lr, "
            f"stronger regularisation (smaller C), or standardizing the features.",
            ConvergenceWarning,
            stacklevel=3,
        )
