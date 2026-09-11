"""K-means on Gaussian blobs: the elbow method, and init-strategy reliability.

K-means needs `n_clusters` chosen up front, so the classic diagnostic is the
"elbow" plot: inertia keeps dropping as `n_clusters` grows (more centroids
can only help), but the drop flattens sharply once `n_clusters` passes the
true number of groups. This script:

1. clusters well-separated blobs and reports how well the recovered
   labels line up with the true generating cluster;
2. sweeps `n_clusters` and prints inertia to show the elbow;
3. compares "k-means++" against "random" init reliability across seeds on
   a deliberately uneven dataset, where a bad random draw is likely.

Run:
    uv run python examples/kmeans.py
"""

from __future__ import annotations

import itertools

import numpy as np

from scratchgrad.cluster import KMeans
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
    """Cluster blobs, sweep `n_clusters` for the elbow, compare init strategies."""
    X, y = make_blobs(
        n_samples=600, n_features=2, centers=4, cluster_std=1.2, random_state=0
    )

    model = KMeans(n_clusters=4, random_state=0).fit(X)
    agreement = _best_permutation_agreement(y, model.labels_, 4)
    print(f"4 well-separated blobs: inertia={model.inertia_:.1f}")
    print(f"  label agreement with the true clusters = {agreement:.3f}")

    print("\nelbow sweep (inertia vs n_clusters)")
    for k in range(1, 8):
        inertia = KMeans(n_clusters=k, random_state=0).fit(X).inertia_
        print(f"  n_clusters={k}  inertia={inertia:9.1f}")

    print("\ninit reliability: one dense cluster + 3 far singleton outliers")
    rng = np.random.default_rng(0)
    dense = rng.normal(0.0, 0.3, size=(100, 2))
    outliers = np.array([[50.0, 50.0], [-50.0, 50.0], [50.0, -50.0]])
    X_uneven = np.vstack([dense, outliers])
    best_inertia = (
        KMeans(n_clusters=4, n_init=50, random_state=0).fit(X_uneven).inertia_
    )

    for init in ("k-means++", "random"):
        hits = sum(
            np.isclose(
                KMeans(n_clusters=4, init=init, n_init=1, random_state=s)
                .fit(X_uneven)
                .inertia_,
                best_inertia,
            )
            for s in range(30)
        )
        print(f"  {init:<10} found the best partition in {hits}/30 single-init runs")


if __name__ == "__main__":
    main()
