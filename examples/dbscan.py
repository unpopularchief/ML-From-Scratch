"""DBSCAN on non-convex moons: the shape K-means cannot solve, plus noise and eps.

K-means partitions space into Voronoi cells around centroids, so it can
never separate two interleaving crescents -- the boundary it needs isn't
a hyperplane. DBSCAN needs no `n_clusters` and follows density instead,
so it recovers non-convex shapes directly. This script:

1. clusters interleaving "moons" and compares against K-means on the same
   data, to make the difference concrete rather than asserted;
2. injects uniform-random outliers into blobs and shows they land in
   `labels_ == -1` (noise) rather than distorting a cluster;
3. sweeps `eps` on one dataset to show the two failure modes: too small
   fragments a real cluster into noise, too large merges separate ones.

Run:
    uv run python examples/dbscan.py
"""

from __future__ import annotations

import itertools

import numpy as np

from scratchgrad.cluster import DBSCAN, KMeans
from scratchgrad.datasets import make_blobs, make_moons


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
    """Cluster moons vs K-means, detect injected noise, sweep eps."""
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=0)

    dbscan_labels = DBSCAN(eps=0.2, min_samples=5).fit(X_moons).labels_
    kmeans_labels = KMeans(n_clusters=2, random_state=0).fit(X_moons).labels_
    dbscan_agreement = _best_permutation_agreement(y_moons, dbscan_labels, 2)
    kmeans_agreement = _best_permutation_agreement(y_moons, kmeans_labels, 2)
    print("interleaving moons: recovering the two true arcs")
    print(f"  DBSCAN  label agreement = {dbscan_agreement:.3f}")
    print(f"  KMeans  label agreement = {kmeans_agreement:.3f}  (no line divides them)")

    print("\nnoise detection: 3 blobs + 15 uniform-random outliers")
    rng = np.random.default_rng(0)
    X_blobs, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.6, random_state=0)
    outliers = rng.uniform(X_blobs.min(), X_blobs.max(), size=(15, 2))
    X_with_noise = np.vstack([X_blobs, outliers])
    model = DBSCAN(eps=0.7, min_samples=5).fit(X_with_noise)
    n_clusters = len(set(model.labels_.tolist()) - {-1})
    n_total_noise = int(np.sum(model.labels_ == -1))
    n_outliers_caught = int(np.sum(model.labels_[-15:] == -1))
    print(
        f"  found {n_clusters} clusters, {n_total_noise} noise points total "
        f"({n_outliers_caught}/15 injected outliers correctly flagged)"
    )

    print("\neps sensitivity on the same 3 blobs (min_samples=5)")
    for eps in (0.15, 0.3, 0.7, 1.5, 3.0):
        labels = DBSCAN(eps=eps, min_samples=5).fit(X_blobs).labels_
        n_clusters = len(set(labels.tolist()) - {-1})
        n_noise = int(np.sum(labels == -1))
        print(f"  eps={eps:<4} -> {n_clusters} clusters, {n_noise:3d} noise points")


if __name__ == "__main__":
    main()
