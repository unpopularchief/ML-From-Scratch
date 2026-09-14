r"""Gaussian Mixture Model fitted by Expectation-Maximization.

Models the data as a weighted sum of ``n_components`` Gaussians and fits it
by maximizing the observed-data log-likelihood

.. math::
    \ell(\pi,\mu,\Sigma) = \sum_{i=1}^n \log \sum_{k=1}^K
        \pi_k\, \mathcal{N}(x_i;\mu_k,\Sigma_k)

which has no closed-form maximizer (a log of a sum). EM instead alternates
two exact steps that provably never decrease :math:`\ell`: an **E-step**
computing each point's posterior responsibility under a component,

.. math::
    \gamma_{ik} = \frac{\pi_k\,\mathcal{N}(x_i;\mu_k,\Sigma_k)}
                       {\sum_j \pi_j\,\mathcal{N}(x_i;\mu_j,\Sigma_j)}

and an **M-step** re-fitting every parameter as a responsibility-weighted
MLE. This is the soft-assignment, shape-and-size-learning generalization of
:class:`scratchgrad.cluster.KMeans` — see ``docs/derivations/gaussian_mixture.md``
§5 for the precise limiting argument.

Full derivation, all four ``covariance_type`` M-step formulas, and the test
plan: ``docs/derivations/gaussian_mixture.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.cluster.kmeans import KMeans
from scratchgrad.typing import FeatureMatrix, FloatArray, IntArray
from scratchgrad.utils.math import logsumexp
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
)

_COVARIANCE_TYPES = ("full", "tied", "diag", "spherical")
_INIT_STRATEGIES = ("kmeans", "random")

# Added to N_k before dividing, matching scikit-learn: guards against a
# component with (numerically) zero responsibility mass dividing by zero.
_NK_EPS = 10.0 * np.finfo(np.float64).eps


# ---------------------------------------------------------------------------
# Log-density, one function per covariance_type (see the derivation doc §9)
# ---------------------------------------------------------------------------


def _cholesky(covariance: FloatArray, reg_covar: float) -> FloatArray:
    """Lower Cholesky factor of a (possibly barely non-PD) covariance matrix.

    Raises
    ------
    ValueError
        If the matrix is not positive-definite even after ``reg_covar`` was
        added to its diagonal by the caller — a clearer error than a raw
        ``numpy.linalg.LinAlgError``, naming the fix.

    """
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "A component's covariance matrix is not positive-definite "
            "(likely too few points, or points nearly collinear, assigned "
            f"to it). Try a larger reg_covar (currently {reg_covar})."
        ) from exc


def _log_gaussian_prob_full(
    X: FeatureMatrix, means: FloatArray, covariances: FloatArray
) -> FloatArray:
    """Full covariance: one Cholesky solve per component, shape ``(n, K)``."""
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    log_prob = np.empty((n_samples, n_components))
    for k in range(n_components):
        chol = _cholesky(covariances[k], reg_covar=0.0)  # already regularised
        log_det = 2.0 * np.sum(np.log(np.diag(chol)))  # log det Σ_k = 2 Σ log L_kk
        diff = X - means[k]  # (n, d)
        z = np.linalg.solve(chol, diff.T)  # L_k z = diffᵀ  ->  z = L_k⁻¹ diffᵀ
        mahalanobis_sq = np.sum(z**2, axis=0)  # ||L_k⁻¹(x_i - μ_k)||²
        log_prob[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + log_det + mahalanobis_sq
        )
    return log_prob


def _log_gaussian_prob_tied(
    X: FeatureMatrix, means: FloatArray, covariance: FloatArray
) -> FloatArray:
    """Tied covariance: one shared Cholesky factor reused for every component."""
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    chol = _cholesky(covariance, reg_covar=0.0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    log_prob = np.empty((n_samples, n_components))
    for k in range(n_components):
        diff = X - means[k]
        z = np.linalg.solve(chol, diff.T)
        mahalanobis_sq = np.sum(z**2, axis=0)
        log_prob[:, k] = -0.5 * (
            n_features * np.log(2.0 * np.pi) + log_det + mahalanobis_sq
        )
    return log_prob


def _log_gaussian_prob_diag(
    X: FeatureMatrix, means: FloatArray, covariances: FloatArray
) -> FloatArray:
    """Diagonal covariance: independent per-feature variances, ``(K, d)``."""
    n_features = X.shape[1]
    precisions = 1.0 / covariances  # (K, d)
    log_det = np.sum(np.log(covariances), axis=1)  # (K,)  Σ_j log σ²_kj
    diff_sq = (X[:, None, :] - means[None, :, :]) ** 2  # (n, K, d)
    mahalanobis = np.sum(diff_sq * precisions[None, :, :], axis=2)  # (n, K)
    return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det[None, :] + mahalanobis)


def _log_gaussian_prob_spherical(
    X: FeatureMatrix, means: FloatArray, covariances: FloatArray
) -> FloatArray:
    """Spherical covariance: one shared scalar variance per component, ``(K,)``."""
    n_features = X.shape[1]
    sq_dist = np.sum((X[:, None, :] - means[None, :, :]) ** 2, axis=2)  # (n, K)
    return -0.5 * (
        n_features * np.log(2.0 * np.pi * covariances)[None, :]
        + sq_dist / covariances[None, :]
    )


def _estimate_log_gaussian_prob(
    X: FeatureMatrix, means: FloatArray, covariances: FloatArray, covariance_type: str
) -> FloatArray:
    r"""Dispatch to the ``covariance_type``-specific log-density, shape ``(n, K)``.

    Each entry ``[i, k]`` is :math:`\log \mathcal N(x_i;\mu_k,\Sigma_k)`.
    """
    if covariance_type == "full":
        return _log_gaussian_prob_full(X, means, covariances)
    if covariance_type == "tied":
        return _log_gaussian_prob_tied(X, means, covariances)
    if covariance_type == "diag":
        return _log_gaussian_prob_diag(X, means, covariances)
    return _log_gaussian_prob_spherical(X, means, covariances)


# ---------------------------------------------------------------------------
# M-step covariance estimators, one per covariance_type
# ---------------------------------------------------------------------------


def _covariances_full(
    X: FeatureMatrix,
    resp: FloatArray,
    means: FloatArray,
    nk: FloatArray,
    reg_covar: float,
) -> FloatArray:
    n_features = X.shape[1]
    n_components = means.shape[0]
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]  # (n, d)
        # Σ_k = (1/N_k) Σ_i γ_ik (x_i-μ_k)(x_i-μ_k)ᵀ
        covariances[k] = (resp[:, k : k + 1] * diff).T @ diff / nk[k]
        covariances[k].flat[:: n_features + 1] += reg_covar  # add to the diagonal
    return covariances


def _covariances_tied(
    X: FeatureMatrix, resp: FloatArray, means: FloatArray, reg_covar: float
) -> FloatArray:
    n_samples, n_features = X.shape
    n_components = means.shape[0]
    covariance = np.zeros((n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariance += (resp[:, k : k + 1] * diff).T @ diff
    covariance /= n_samples  # pooled scatter, normalised by ALL samples
    covariance.flat[:: n_features + 1] += reg_covar
    return covariance


def _covariances_diag(
    X: FeatureMatrix,
    resp: FloatArray,
    means: FloatArray,
    nk: FloatArray,
    reg_covar: float,
) -> FloatArray:
    # Centred two-pass form (not the raw-moment E[x²]-E[x]² shortcut) —
    # avoids the catastrophic-cancellation gotcha documented for
    # DecisionTreeRegressor's variance sums on offset/large-baseline data.
    diff_sq = (X[:, None, :] - means[None, :, :]) ** 2  # (n, K, d)
    covariances = np.sum(resp[:, :, None] * diff_sq, axis=0) / nk[:, None]  # (K, d)
    return covariances + reg_covar


def _covariances_spherical(
    X: FeatureMatrix,
    resp: FloatArray,
    means: FloatArray,
    nk: FloatArray,
    reg_covar: float,
) -> FloatArray:
    n_features = X.shape[1]
    sq_dist = np.sum((X[:, None, :] - means[None, :, :]) ** 2, axis=2)  # (n, K)
    covariances = np.sum(resp * sq_dist, axis=0) / (nk * n_features)  # (K,)
    return covariances + reg_covar


def _estimate_covariances(
    X: FeatureMatrix,
    resp: FloatArray,
    means: FloatArray,
    nk: FloatArray,
    covariance_type: str,
    reg_covar: float,
) -> FloatArray:
    """Dispatch to the ``covariance_type``-specific weighted-MLE covariance."""
    if covariance_type == "full":
        return _covariances_full(X, resp, means, nk, reg_covar)
    if covariance_type == "tied":
        return _covariances_tied(X, resp, means, reg_covar)
    if covariance_type == "diag":
        return _covariances_diag(X, resp, means, nk, reg_covar)
    return _covariances_spherical(X, resp, means, nk, reg_covar)


# ---------------------------------------------------------------------------
# E-step / M-step / the EM loop
# ---------------------------------------------------------------------------


def _e_step(
    X: FeatureMatrix,
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    covariance_type: str,
) -> tuple[FloatArray, FloatArray]:
    r"""One E-step: responsibilities plus the per-point log-evidence.

    Returns ``(log_prob_norm, log_resp)`` — ``log_prob_norm`` is
    :math:`\log p(x_i)`, shape ``(n,)``; ``log_resp`` is
    :math:`\log \gamma_{ik}`, shape ``(n, K)``.
    """
    log_prob = _estimate_log_gaussian_prob(X, means, covariances, covariance_type)
    weighted_log_prob = log_prob + np.log(weights)  # log[π_k N(x;μ_k,Σ_k)]
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)  # log p(x_i)
    log_resp = weighted_log_prob - log_prob_norm[:, None]  # log γ_ik
    return log_prob_norm, log_resp


def _m_step(
    X: FeatureMatrix, resp: FloatArray, covariance_type: str, reg_covar: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One M-step: weighted MLE of weights, means, covariances from ``resp``."""
    n_samples = X.shape[0]
    nk = resp.sum(axis=0) + _NK_EPS  # N_k
    means = (resp.T @ X) / nk[:, None]  # μ_k
    covariances = _estimate_covariances(X, resp, means, nk, covariance_type, reg_covar)
    weights = nk / n_samples  # π_k
    return weights, means, covariances


def _init_random_resp(
    n_samples: int, n_components: int, rng: np.random.Generator
) -> FloatArray:
    """Uniform-random initial responsibilities, each row normalised to sum to 1."""
    resp = rng.uniform(size=(n_samples, n_components))
    resp /= resp.sum(axis=1, keepdims=True)
    return resp


def _init_kmeans_resp(
    X: FeatureMatrix, n_components: int, rng: np.random.Generator
) -> FloatArray:
    """One-hot initial responsibilities from a single KMeans run's hard labels."""
    labels = KMeans(n_clusters=n_components, n_init=1, random_state=rng).fit(X).labels_
    resp = np.zeros((X.shape[0], n_components))
    resp[np.arange(X.shape[0]), labels] = 1.0
    return resp


def _em(
    X: FeatureMatrix,
    weights: FloatArray,
    means: FloatArray,
    covariances: FloatArray,
    covariance_type: str,
    max_iter: int,
    tol: float,
    reg_covar: float,
) -> tuple[FloatArray, FloatArray, FloatArray, float, int, bool]:
    """Run EM to convergence (or ``max_iter``) from an initial M-step result.

    Returns ``(weights, means, covariances, lower_bound, n_iter, converged)``
    — see ``docs/derivations/gaussian_mixture.md`` §4/§8.
    """
    lower_bound = -np.inf
    converged = False
    for _iteration in range(1, max_iter + 1):
        log_prob_norm, log_resp = _e_step(
            X, weights, means, covariances, covariance_type
        )
        new_lower_bound = float(log_prob_norm.mean())  # avg per-sample log-likelihood
        weights, means, covariances = _m_step(
            X, np.exp(log_resp), covariance_type, reg_covar
        )
        if new_lower_bound - lower_bound < tol:
            lower_bound = new_lower_bound
            converged = True
            break
        lower_bound = new_lower_bound
    return weights, means, covariances, lower_bound, _iteration, converged


def _n_parameters(n_components: int, n_features: int, covariance_type: str) -> int:
    """Free-parameter count for ``bic``/``aic`` — see the derivation doc §10."""
    mean_params = n_components * n_features
    weight_params = n_components - 1  # simplex constraint removes one d.o.f.
    if covariance_type == "full":
        cov_params = n_components * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        cov_params = n_features * (n_features + 1) // 2
    elif covariance_type == "diag":
        cov_params = n_components * n_features
    else:  # spherical
        cov_params = n_components
    return mean_params + weight_params + cov_params


def _sample_component(
    mean: FloatArray, chol: FloatArray, n_samples: int, rng: np.random.Generator
) -> FloatArray:
    r"""Draw ``n_samples`` points from :math:`\mathcal N(\text{mean}, LL^\top)`."""
    n_features = mean.shape[0]
    z = rng.standard_normal((n_samples, n_features))
    return mean + z @ chol.T


class GaussianMixture(Estimator):
    r"""Gaussian mixture model fitted by Expectation-Maximization.

    Parameters
    ----------
    n_components : int, default=1
        Number of mixture components ``K``. Must be between 1 and
        ``n_samples``.
    covariance_type : {"full", "tied", "diag", "spherical"}, default="full"
        Structure imposed on every component's covariance — see
        ``docs/derivations/gaussian_mixture.md`` §9.
    tol : float, default=1e-3
        Stop a run once the average per-sample log-likelihood improves by
        less than ``tol`` between iterations.
    reg_covar : float, default=1e-6
        Added to the diagonal of every covariance after each M-step, the
        GaussianMixture analogue of ``GaussianNB``'s ``var_smoothing``
        floor — guards against a singular covariance from a component with
        too few or too-collinear points.
    max_iter : int, default=100
        Maximum EM iterations per run.
    n_init : int, default=1
        Number of independent random restarts; the run with the highest
        final average log-likelihood (``lower_bound_``) is kept.
    init_params : {"kmeans", "random"}, default="kmeans"
        Initial-responsibility strategy. ``"kmeans"`` runs this project's
        own :class:`~scratchgrad.cluster.KMeans` once and one-hot encodes
        its labels (§6); ``"random"`` draws uniform-random responsibilities.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds initialization. See
        :func:`scratchgrad.utils.validation.check_random_state`.

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
        Mixing weights :math:`\pi_k` of the kept run.
    means_ : ndarray of shape (n_components, n_features)
        Component means :math:`\mu_k`.
    covariances_ : ndarray
        Component covariances :math:`\Sigma_k`; shape depends on
        ``covariance_type`` — ``(K, d, d)`` for ``"full"``, ``(d, d)`` for
        ``"tied"``, ``(K, d)`` for ``"diag"``, ``(K,)`` for ``"spherical"``.
    labels_ : ndarray of shape (n_samples,), dtype int64
        Hard assignment (``argmax`` responsibility) of each training row
        under the final fitted parameters — not part of scikit-learn's
        public API, included here for consistency with
        :class:`~scratchgrad.cluster.KMeans`/:class:`~scratchgrad.cluster.DBSCAN`.
    lower_bound_ : float
        Average per-sample log-likelihood of the kept run at convergence
        (or at ``max_iter``) — the objective EM maximizes (§4).
    n_iter_ : int
        Number of EM iterations run by the kept run.
    converged_ : bool
        Whether the kept run's stopping condition was ``tol``, not
        ``max_iter``.

    Notes
    -----
    See ``docs/derivations/gaussian_mixture.md`` for the full EM derivation,
    the KMeans limiting case (§5), and each covariance type's closed-form
    M-step (§9).

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.cluster import GaussianMixture
    >>> X = np.array([[0.0, 0.0], [0.2, -0.1], [10.0, 10.0], [10.1, 9.9]])
    >>> model = GaussianMixture(n_components=2, random_state=0).fit(X)
    >>> model.predict(np.array([[0.1, 0.1], [9.9, 10.0]])).tolist()
    [1, 0]

    """

    def __init__(
        self,
        n_components: int = 1,
        covariance_type: str = "full",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X: FeatureMatrix, y: object = None) -> GaussianMixture:
        """Fit the mixture to ``X`` by EM.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to fit.
        y : ignored
            Present for API consistency (``fit(X, y=None)``) — unsupervised.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``n_components``, ``n_init``, or ``max_iter`` is not
            positive, ``tol``/``reg_covar`` is negative, ``n_components``
            exceeds the number of samples, or ``covariance_type``/
            ``init_params`` is not a recognised value.

        """
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}.")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}.")
        if self.tol < 0.0:
            raise ValueError(f"tol must be >= 0, got {self.tol}.")
        if self.reg_covar < 0.0:
            raise ValueError(f"reg_covar must be >= 0, got {self.reg_covar}.")
        if self.covariance_type not in _COVARIANCE_TYPES:
            raise ValueError(
                f"covariance_type must be one of {_COVARIANCE_TYPES}, got "
                f"{self.covariance_type!r}."
            )
        if self.init_params not in _INIT_STRATEGIES:
            raise ValueError(
                f"init_params must be one of {_INIT_STRATEGIES}, got "
                f"{self.init_params!r}."
            )

        X = check_array(X)
        if self.n_components > X.shape[0]:
            raise ValueError(
                f"n_components={self.n_components} is larger than the number "
                f"of samples ({X.shape[0]})."
            )

        rng = check_random_state(self.random_state)

        best: tuple[FloatArray, FloatArray, FloatArray, float, int, bool] | None = None
        for _ in range(self.n_init):
            if self.init_params == "kmeans":
                resp0 = _init_kmeans_resp(X, self.n_components, rng)
            else:
                resp0 = _init_random_resp(X.shape[0], self.n_components, rng)
            weights, means, covariances = _m_step(
                X, resp0, self.covariance_type, self.reg_covar
            )
            result = _em(
                X,
                weights,
                means,
                covariances,
                self.covariance_type,
                self.max_iter,
                self.tol,
                self.reg_covar,
            )
            if best is None or result[3] > best[3]:
                best = result

        (
            self.weights_,
            self.means_,
            self.covariances_,
            self.lower_bound_,
            self.n_iter_,
            self.converged_,
        ) = best
        # Re-run the E-step against the KEPT run's parameters — when
        # n_init > 1 the best run need not be the last one computed, so
        # `labels_` must be re-synced to whichever parameters were kept.
        _, log_resp = self._e_step(X)
        self.labels_ = np.argmax(log_resp, axis=1)
        return self

    def predict(self, X: FeatureMatrix) -> IntArray:
        """Return the most likely component index for each row of ``X``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        _, log_resp = self._e_step(X)
        return np.argmax(log_resp, axis=1)

    def predict_proba(self, X: FeatureMatrix) -> FloatArray:
        r"""Return the responsibility of every component for each row of ``X``.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Row ``i`` sums to 1: :math:`\gamma_{i1},\dots,\gamma_{iK}`.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        _, log_resp = self._e_step(X)
        return np.exp(log_resp)

    def fit_predict(self, X: FeatureMatrix, y: object = None) -> IntArray:
        """Equivalent to ``fit(X).labels_`` — fit and return the training labels."""
        return self.fit(X).labels_

    def score_samples(self, X: FeatureMatrix) -> FloatArray:
        r"""Return the log-likelihood :math:`\log p(x_i)` of every row of ``X``.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        log_prob_norm, _ = self._e_step(X)
        return log_prob_norm

    def score(self, X: FeatureMatrix, y: object = None) -> float:
        """Return the average per-sample log-likelihood of ``X``.

        The same quantity :meth:`fit` maximizes (``lower_bound_`` is this
        value on the training data at convergence).
        """
        return float(self.score_samples(X).mean())

    def bic(self, X: FeatureMatrix) -> float:
        """Bayesian Information Criterion of ``X`` under the fitted mixture.

        ``-2 * n * score(X) + n_parameters * log(n)``, lower is better. See
        ``docs/derivations/gaussian_mixture.md`` §10.
        """
        check_is_fitted(self, "weights_")
        n_samples = X.shape[0]
        n_features = self.means_.shape[1]
        n_params = _n_parameters(self.n_components, n_features, self.covariance_type)
        return -2.0 * n_samples * self.score(X) + n_params * np.log(n_samples)

    def aic(self, X: FeatureMatrix) -> float:
        """Akaike Information Criterion of ``X`` under the fitted mixture.

        ``-2 * n * score(X) + 2 * n_parameters``, lower is better. See
        ``docs/derivations/gaussian_mixture.md`` §10.
        """
        check_is_fitted(self, "weights_")
        n_samples = X.shape[0]
        n_features = self.means_.shape[1]
        n_params = _n_parameters(self.n_components, n_features, self.covariance_type)
        return -2.0 * n_samples * self.score(X) + 2.0 * n_params

    def sample(self, n_samples: int = 1) -> tuple[FloatArray, IntArray]:
        """Draw ``n_samples`` points from the fitted mixture.

        Parameters
        ----------
        n_samples : int, default=1
            Number of points to generate.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Generated points.
        y : ndarray of shape (n_samples,), dtype int64
            Generating component index of each row.

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``n_samples < 1``.

        """
        check_is_fitted(self, "weights_")
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}.")

        rng = check_random_state(self.random_state)
        counts = rng.multinomial(n_samples, self.weights_)

        if self.covariance_type == "tied":
            tied_chol = _cholesky(self.covariances_, reg_covar=self.reg_covar)

        X_parts = []
        y_parts = []
        for k, count in enumerate(counts):
            if count == 0:
                continue
            if self.covariance_type == "full":
                chol = _cholesky(self.covariances_[k], reg_covar=self.reg_covar)
            elif self.covariance_type == "tied":
                chol = tied_chol
            elif self.covariance_type == "diag":
                chol = np.diag(np.sqrt(self.covariances_[k]))
            else:  # spherical
                n_features = self.means_.shape[1]
                chol = np.sqrt(self.covariances_[k]) * np.eye(n_features)
            X_parts.append(_sample_component(self.means_[k], chol, int(count), rng))
            y_parts.append(np.full(int(count), k, dtype=np.int64))

        return np.vstack(X_parts), np.concatenate(y_parts)

    def _e_step(self, X: FeatureMatrix) -> tuple[FloatArray, FloatArray]:
        """Validate ``X`` against the fitted model, then run the E-step."""
        check_is_fitted(self, "weights_")
        X = check_array(X)
        if X.shape[1] != self.means_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.means_.shape[1]}."
            )
        return _e_step(
            X, self.weights_, self.means_, self.covariances_, self.covariance_type
        )
