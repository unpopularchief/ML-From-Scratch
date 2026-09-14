"""PCA on correlated data: variance captured, and reconstruction error.

PCA's whole pitch is "a few directions capture most of the variance." This
script:

1. builds a strongly-correlated 2D Gaussian blob and shows that a single
   component already captures almost all of its variance;
2. projects a higher-dimensional, structured dataset down to 2 components
   and reports how much reconstruction error that costs versus keeping
   every component (which is always exact, by construction -- see
   docs/derivations/pca.md section 2);
3. cross-checks the SVD-based `fit` against the from-scratch power-iteration
   implementation kept for exactly this purpose (derivation section 6).

Run:
    uv run python examples/pca.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.decomposition import PCA
from scratchgrad.decomposition.pca import _power_iteration_pca


def main() -> None:
    """Fit PCA on correlated data and report captured/reconstructed variance."""
    rng = np.random.default_rng(0)

    # Strongly correlated 2D blob: most of the spread lies along one axis.
    raw = rng.normal(size=(300, 2)) * np.array([3.0, 0.2])
    theta = np.pi / 4
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    X_2d = raw @ rotation.T

    model_2d = PCA(n_components=1).fit(X_2d)
    ratio_2d = model_2d.explained_variance_ratio_[0]
    print("2D correlated blob:")
    print(f"  1 component captures {ratio_2d:.4f} of variance")

    # Higher-dimensional structured data (built from a low-rank signal plus
    # noise), projected down to 2 components.
    n_samples, n_features, true_rank = 400, 8, 2
    signal = rng.normal(size=(n_samples, true_rank)) @ rng.normal(
        size=(true_rank, n_features)
    )
    X_hd = signal + 0.1 * rng.normal(size=(n_samples, n_features))

    full = PCA(n_components=n_features).fit(X_hd)
    reduced = PCA(n_components=2).fit(X_hd)
    reconstructed = reduced.inverse_transform(reduced.transform(X_hd))
    reconstruction_error = np.mean(np.sum((X_hd - reconstructed) ** 2, axis=1))
    ratios = full.explained_variance_ratio_.round(4)
    captured = reduced.explained_variance_ratio_.sum()

    print(f"\n{n_features}D data built from a rank-{true_rank} signal + noise:")
    print(f"  explained_variance_ratio_ (all components) = {ratios}")
    print(f"  2 components capture {captured:.4f} of variance")
    print(f"  2 components' reconstruction MSE = {reconstruction_error:.4f}")

    # Cross-check: the from-scratch power-iteration path (derivation §6)
    # should recover the same components/eigenvalues as the SVD-based fit.
    components, eigenvalues = _power_iteration_pca(X_hd, 2, np.random.default_rng(1))
    for i in range(2):
        if np.dot(components[i], reduced.components_[i]) < 0:
            components[i] *= -1
    max_component_diff = np.max(np.abs(components - reduced.components_))
    max_eigenvalue_diff = np.max(np.abs(eigenvalues - reduced.explained_variance_))
    print(
        f"\npower iteration vs SVD: max component diff = {max_component_diff:.2e}, "
        f"max eigenvalue diff = {max_eigenvalue_diff:.2e}"
    )


if __name__ == "__main__":
    main()
