"""scikit-learn parity for KMeans.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

Unlike the tree ensembles, K-means with an **explicit `init` array** has no
RNG anywhere in Lloyd's algorithm (assignment and update are both
deterministic argmins), so this is the KMeans analogue of the tree's
`max_features=None` deterministic path: exact parity, not just a
tolerance. That includes an exercised empty-cluster relocation, since
`sklearn.cluster.KMeans` relocates an empty cluster's centroid the same
way (steal the point farthest from its own cluster's centroid, and
exclude it from that cluster's mean).

`init="k-means++"` / `"random"` draw from our own `Generator`, which does
not share a stream with sklearn's RNG, so those are tolerance-only:
comparable inertia across several seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.cluster import KMeans
from scratchgrad.datasets import make_blobs

pytestmark = pytest.mark.reference


def test_explicit_init_matches_sklearn_exactly() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    X, _ = make_blobs(
        n_samples=200, n_features=3, centers=4, cluster_std=2.0, random_state=0
    )
    init = X[[0, 50, 100, 150]].copy()

    ours = KMeans(n_clusters=4, init=init, n_init=10, random_state=0).fit(X)
    theirs = cluster.KMeans(
        n_clusters=4, init=init, n_init=1, algorithm="lloyd", random_state=0
    ).fit(X)

    np.testing.assert_allclose(ours.cluster_centers_, theirs.cluster_centers_)
    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    assert ours.inertia_ == pytest.approx(theirs.inertia_)
    assert ours.n_iter_ == theirs.n_iter_


def test_explicit_init_with_an_empty_cluster_matches_sklearn_exactly() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.normal(0, 0.1, size=(20, 2)), rng.normal([20, 20], 0.1, size=(20, 2))]
    )
    # third seed is nowhere near any data -> guaranteed empty after the
    # first assignment step, exercising the relocation rule on both sides.
    init = np.array([[0.0, 0.0], [20.0, 20.0], [1000.0, 1000.0]])

    ours = KMeans(n_clusters=3, init=init, n_init=1, random_state=0).fit(X)
    theirs = cluster.KMeans(
        n_clusters=3, init=init, n_init=1, algorithm="lloyd", random_state=0
    ).fit(X)

    np.testing.assert_allclose(ours.cluster_centers_, theirs.cluster_centers_)
    np.testing.assert_array_equal(ours.labels_, theirs.labels_)
    assert ours.n_iter_ == theirs.n_iter_


def test_kmeans_plusplus_inertia_tracks_sklearn() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    X, _ = make_blobs(
        n_samples=400, n_features=4, centers=5, cluster_std=3.0, random_state=1
    )

    ours = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)
    theirs = cluster.KMeans(n_clusters=5, n_init=10, random_state=0).fit(X)

    assert abs(ours.inertia_ - theirs.inertia_) / theirs.inertia_ < 0.05


def test_predict_on_held_out_data_tracks_sklearn_inertia() -> None:
    cluster = pytest.importorskip("sklearn.cluster")

    X, _ = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=0)
    X_test, _ = make_blobs(n_samples=100, centers=3, cluster_std=2.0, random_state=1)

    ours = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    theirs = cluster.KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)

    ours_dist = ours.transform(X_test)
    ours_inertia = np.sum(np.min(ours_dist, axis=1) ** 2)
    theirs_inertia = -theirs.score(X_test)

    assert abs(ours_inertia - theirs_inertia) / theirs_inertia < 0.05
