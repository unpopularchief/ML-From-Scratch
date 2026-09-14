"""GaussianMixture on Gaussian blobs: covariance types, BIC, and generation.

Unlike `KMeans`, `GaussianMixture` fits a real generative model — it can
report a log-likelihood, compare `covariance_type`s and `n_components` via
BIC/AIC, and draw new synthetic points. This script:

1. fits all four `covariance_type`s to the same blobs and compares them to
   `KMeans` on the same data (docs/derivations/gaussian_mixture.md sec 5);
2. sweeps `n_components` and reports BIC, the standard way to pick K for a
   mixture model;
3. draws samples from a fitted mixture and checks the generated points
   land back near their true component's mean.

Run:
    uv run python examples/gaussian_mixture.py
"""

from __future__ import annotations

import itertools

import numpy as np

from scratchgrad.cluster import GaussianMixture, KMeans
from scratchgrad.datasets import make_blobs


def _best_permutation_agreement(
    true: np.ndarray, pred: np.ndarray, n_classes: int
) -> float:
    """Best label-agreement fraction over every relabelling of `pred`."""
    best = 0.0
    for perm in itertools.permutations(range(n_classes)):
        remapped = np.array(perm)[pred]
        best = max(best, float(np.mean(remapped == true)))
    return best


def main() -> None:
    """Fit every covariance_type, sweep n_components via BIC, then sample."""
    X, y = make_blobs(
        n_samples=600, n_features=2, centers=4, cluster_std=1.2, random_state=0
    )

    print("covariance_type comparison (n_components=4, n_init=5)")
    kmeans_labels = KMeans(n_clusters=4, random_state=0).fit(X).labels_
    for covariance_type in ("full", "tied", "diag", "spherical"):
        model = GaussianMixture(
            n_components=4, covariance_type=covariance_type, n_init=5, random_state=0
        ).fit(X)
        true_agreement = _best_permutation_agreement(y, model.labels_, 4)
        kmeans_agreement = _best_permutation_agreement(kmeans_labels, model.labels_, 4)
        print(
            f"  {covariance_type:<10} converged={model.converged_!s:<5} "
            f"n_iter={model.n_iter_:<3} lower_bound={model.lower_bound_:8.3f}  "
            f"agreement(true)={true_agreement:.3f}  "
            f"agreement(KMeans)={kmeans_agreement:.3f}"
        )

    print("\nBIC sweep (choosing n_components, covariance_type='full')")
    for k in range(1, 8):
        model = GaussianMixture(n_components=k, n_init=5, random_state=0).fit(X)
        print(f"  n_components={k}  bic={model.bic(X):10.1f}  aic={model.aic(X):10.1f}")

    print("\ngenerative check: sample from a fitted 4-component mixture")
    model = GaussianMixture(n_components=4, n_init=5, random_state=0).fit(X)
    samples, sample_labels = model.sample(2000)
    for k in range(4):
        drawn = samples[sample_labels == k]
        drift = np.linalg.norm(drawn.mean(axis=0) - model.means_[k])
        print(
            f"  component {k}: drew {drawn.shape[0]:4d} points, "
            f"empirical-mean drift from means_[{k}] = {drift:.3f}"
        )


if __name__ == "__main__":
    main()
