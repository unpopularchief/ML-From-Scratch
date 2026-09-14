r"""Principal Component Analysis.

Finds the ``n_components`` orthonormal directions that maximise the
variance of the projected data — equivalently (see
``docs/derivations/pca.md`` §2), the directions minimising squared
reconstruction error:

.. math::
    \max_{w:\, \lVert w \rVert = 1} \mathrm{Var}(X_c w) = \max_w w^\top C w,
    \qquad C = \frac{1}{n-1} X_c^\top X_c

Setting the Lagrangian's gradient to zero gives :math:`Cw = \lambda w`:
the maximising directions are **eigenvectors of the covariance matrix**,
ordered by eigenvalue. Computed via the SVD of the centered data
:math:`X_c = U \Sigma V^\top` rather than an explicit eigendecomposition
of :math:`C` (squaring the condition number) — :math:`C = V \frac{\Sigma^2}{n-1}
V^\top` reads the same eigenvectors/eigenvalues straight off the SVD.

Full walkthrough: ``docs/derivations/pca.md``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.typing import FeatureMatrix, FloatArray
from scratchgrad.utils.validation import check_array, check_is_fitted


def _fix_signs(U: FloatArray, Vt: FloatArray) -> None:
    """Deterministic eigenvector sign, in place: match scikit-learn's convention.

    :math:`v` and :math:`-v` are both valid eigenvectors of the same
    eigenvalue, and ``np.linalg.svd`` makes no promise about which one it
    returns. For each row of ``Vt``, if its largest-magnitude entry is
    negative, flip the row (and the matching column of ``U``, so
    ``U @ diag(S) @ Vt`` still reconstructs the original data). See
    ``docs/derivations/pca.md`` §5.
    """
    for i in range(Vt.shape[0]):
        largest = np.argmax(np.abs(Vt[i]))
        if Vt[i, largest] < 0:
            Vt[i] *= -1.0
            U[:, i] *= -1.0


def _power_iteration_pca(
    X: FeatureMatrix,
    n_components: int,
    rng: np.random.Generator,
    n_iter: int = 500,
    tol: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    r"""Top ``n_components`` eigenvectors/eigenvalues of ``cov(X)`` via power iteration.

    An independent algorithm from the SVD path in :meth:`PCA.fit`, kept
    only to be tested against it (``plan.md`` §6: "ship a from-scratch
    version... alongside the ``np.linalg`` one"). Power iteration repeatedly
    applies the covariance matrix to a random vector and renormalises,
    which converges to the top eigenvector because each application scales
    the dominant eigen-component by :math:`\lambda_1` and every other
    component by a strictly smaller :math:`\lambda_j`. Deflation
    (subtracting :math:`\lambda_1 w_1 w_1^\top` from :math:`C`) zeroes the
    top eigenvalue while leaving every other eigenpair untouched, so
    repeating finds the next component. See ``docs/derivations/pca.md`` §6.

    Returns
    -------
    components : ndarray of shape (n_components, n_features)
    eigenvalues : ndarray of shape (n_components,)

    """
    n_samples, n_features = X.shape
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n_samples - 1)  # C = (1/(n-1)) X_c^T X_c

    components = np.empty((n_components, n_features))
    eigenvalues = np.empty(n_components)
    for k in range(n_components):
        v = rng.normal(size=n_features)
        v /= np.linalg.norm(v)
        for _ in range(n_iter):
            new_v = cov @ v
            norm = np.linalg.norm(new_v)
            new_v /= norm
            if np.linalg.norm(new_v - v) < tol or np.linalg.norm(new_v + v) < tol:
                v = new_v
                break
            v = new_v
        eigenvalue = float(
            v @ cov @ v
        )  # Rayleigh quotient at the converged eigenvector
        components[k] = v
        eigenvalues[k] = eigenvalue
        cov = cov - eigenvalue * np.outer(v, v)  # deflation: zero out this eigenpair
    return components, eigenvalues


class PCA(Estimator):
    r"""Project data onto its top ``n_components`` directions of variance.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components to keep. ``None`` keeps
        ``min(n_samples, n_features)`` (all of them — a change of basis
        with no dimensionality reduction).

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal directions (unit-norm rows), sorted by
        ``explained_variance_`` descending. Sign-fixed for determinism —
        see ``docs/derivations/pca.md`` §5.
    explained_variance_ : ndarray of shape (n_components,)
        Variance of the data along each component, i.e. the corresponding
        eigenvalue of the covariance matrix.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        ``explained_variance_`` divided by the *total* variance across all
        ``min(n_samples, n_features)`` components (not just the kept
        ``n_components``), so it under-counts to less than 1 when
        components are dropped.
    singular_values_ : ndarray of shape (n_components,)
        Singular values of the centered data corresponding to each
        component.
    mean_ : ndarray of shape (n_features,)
        Per-feature mean subtracted before projecting.
    n_components_ : int
        The effective ``n_components`` used (resolved from ``None``).

    Notes
    -----
    Computed via the SVD of the centered data rather than an explicit
    eigendecomposition of the covariance matrix, for numerical stability
    (``docs/derivations/pca.md`` §4). See that file for the full
    variance-maximisation derivation and the reconstruction-error
    equivalence.

    Examples
    --------
    >>> import numpy as np
    >>> from scratchgrad.decomposition import PCA
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    >>> model = PCA(n_components=1).fit(X)
    >>> model.explained_variance_ratio_[0] > 0.99
    np.True_

    """

    def __init__(self, n_components: int | None = None) -> None:
        """See the class docstring for parameter descriptions."""
        self.n_components = n_components

    def fit(self, X: FeatureMatrix, y: object = None) -> PCA:
        """Compute the principal components of ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to fit.
        y : ignored
            Present for API consistency (``fit(X, y=None)``) — PCA is
            unsupervised.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If ``n_components`` is not a positive integer, or exceeds
            ``min(n_samples, n_features)``.

        """
        X = check_array(X)
        n_samples, n_features = X.shape
        max_components = min(n_samples, n_features)

        n_components = (
            max_components if self.n_components is None else self.n_components
        )
        if n_components < 1 or n_components > max_components:
            raise ValueError(
                f"n_components must be between 1 and min(n_samples, n_features)"
                f"={max_components}, got {n_components}."
            )

        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Full SVD of the centered data, not eigh(cov) -- see derivation §4.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        _fix_signs(U, Vt)

        # All max_components eigenvalues are needed for the ratio's
        # denominator (total variance) even though only the top
        # n_components are kept in the public attributes.
        all_explained_variance = (S**2) / (n_samples - 1)
        total_var = all_explained_variance.sum()

        self.n_components_ = n_components
        self.components_ = Vt[:n_components]
        self.singular_values_ = S[:n_components]
        self.explained_variance_ = all_explained_variance[:n_components]
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total_var
            if total_var > 0.0
            else np.zeros(n_components)
        )
        return self

    def transform(self, X: FeatureMatrix) -> FloatArray:
        """Project ``X`` onto the fitted principal components.

        Returns
        -------
        ndarray of shape (n_samples, n_components)

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X`` has a different number of features than training.

        """
        check_is_fitted(self, "components_")
        X = check_array(X)
        self._check_n_features(X)
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: FeatureMatrix, y: object = None) -> FloatArray:
        """Equivalent to ``fit(X).transform(X)``."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: FloatArray) -> FloatArray:
        """Map projected data back into the original feature space.

        Exact only when ``n_components == min(n_samples, n_features)`` at
        fit time; otherwise this is the reconstruction from the kept
        components, with the discarded variance lost (§2's residual term).

        Raises
        ------
        NotFittedError
            If called before :meth:`fit`.
        ValueError
            If ``X_transformed`` doesn't have ``n_components_`` columns.

        """
        check_is_fitted(self, "components_")
        X_transformed = check_array(X_transformed)
        if X_transformed.shape[1] != self.n_components_:
            raise ValueError(
                f"X_transformed has {X_transformed.shape[1]} features, but this "
                f"model has n_components_={self.n_components_}."
            )
        return X_transformed @ self.components_ + self.mean_

    def score(self, X: FeatureMatrix, y: object = None) -> float:
        """Return the negative mean squared reconstruction error on ``X``.

        Higher (closer to 0) is better, following the project's "higher is
        better" `score` convention. This is the direct reconstruction-error
        form of the objective derived in §2 -- not scikit-learn's
        probabilistic-PCA log-likelihood, which is a different model.
        """
        X = check_array(X)
        self._check_n_features(X)
        reconstructed = self.inverse_transform(self.transform(X))
        return -float(np.mean(np.sum((X - reconstructed) ** 2, axis=1)))

    def _check_n_features(self, X: FeatureMatrix) -> None:
        """Raise ``ValueError`` if ``X`` doesn't match the fitted feature count."""
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"X has {X.shape[1]} features, but this model was fitted with "
                f"{self.mean_.shape[0]}."
            )
