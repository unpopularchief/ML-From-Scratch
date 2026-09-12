"""Tests for scratchgrad.cluster.DBSCAN.

Tiers (plan.md section 3): DBSCAN has no objective and no gradient — it
*defines* clusters combinatorially (docs/derivations/dbscan.md section 2),
so the correctness analogues are (a) hand-verified core/border/noise
labelling on tiny constructed data and (b) an independently-structured
brute-force reachability computation (a different traversal order than the
module's own stack-based expansion) agreeing with the fitted partition.
Plus correctness invariants that follow directly from the definition,
metric sensitivity, the fit contract, behavioral recovery of a non-convex
shape KMeans cannot solve, and edge cases.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scratchgrad.cluster import DBSCAN
from scratchgrad.cluster.dbscan import _core_points, _expand_clusters
from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.metrics.pairwise import euclidean_distance


def _brute_force_partition(
    X: np.ndarray, eps: float, min_samples: int
) -> tuple[list[set[int]], set[int]]:
    """Independently-structured reference: nested loops + a BFS queue.

    Deliberately not reusing `_core_points`/`_expand_clusters` (a plain
    Python double loop for the neighbor graph, a FIFO queue instead of the
    module's LIFO stack) so this stands in for a gradient check on an
    algorithm with no gradient. Returns (list of clusters as index sets,
    noise index set) -- comparable up to cluster ordering, which is
    arbitrary.
    """
    n = X.shape[0]
    neighbors = []
    for i in range(n):
        row = []
        for j in range(n):
            d = float(np.sqrt(np.sum((X[i] - X[j]) ** 2)))
            if d <= eps:
                row.append(j)
        neighbors.append(row)
    is_core = [len(neighbors[i]) >= min_samples for i in range(n)]

    assigned = [False] * n
    clusters: list[set[int]] = []
    for seed in range(n):
        if assigned[seed] or not is_core[seed]:
            continue
        cluster: set[int] = set()
        queue = [seed]
        assigned[seed] = True
        while queue:
            p = queue.pop(0)
            cluster.add(p)
            if is_core[p]:
                for q in neighbors[p]:
                    if q not in cluster:
                        cluster.add(q)
                        if not assigned[q]:
                            assigned[q] = True
                            queue.append(q)
        clusters.append(cluster)
    clustered = set().union(*clusters) if clusters else set()
    noise = set(range(n)) - clustered
    return clusters, noise


def _best_permutation_agreement(
    true: np.ndarray, pred: np.ndarray, n_classes: int
) -> float:
    """Best agreement fraction over every relabelling (cluster labels are arbitrary)."""
    best = 0.0
    for perm in itertools.permutations(range(n_classes)):
        remapped = np.array(perm)[pred]
        best = max(best, np.mean(remapped == true))
    return best


class TestAnalytic:
    def test_chain_of_close_points_forms_one_cluster(self) -> None:
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        model = DBSCAN(eps=1.5, min_samples=2).fit(X)
        assert set(model.labels_.tolist()) == {0}

    def test_isolated_point_below_min_samples_is_noise(self) -> None:
        X = np.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.1], [100.0, 100.0]])
        model = DBSCAN(eps=0.5, min_samples=3).fit(X)
        assert model.labels_[3] == -1
        assert model.labels_[0] == model.labels_[1] == model.labels_[2] != -1

    def test_hand_verified_core_border_noise(self) -> None:
        # 0,1,2,3 form a tight square, all mutually within eps of each
        # other -- each has exactly 4 neighbors (itself + 3 others), so
        # with min_samples=4 all four are core. 4 is within eps of point 1
        # only (distance 0.3), and has just 2 neighbors of its own (itself
        # + point 1) -- below min_samples, so it's a border point pulled
        # into the same cluster. 5 is far from everything: noise.
        X = np.array(
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [0.0, 0.2],
                [0.2, 0.2],
                [0.5, 0.0],
                [10.0, 10.0],
            ]
        )
        model = DBSCAN(eps=0.3, min_samples=4).fit(X)
        np.testing.assert_array_equal(sorted(model.core_sample_indices_), [0, 1, 2, 3])
        assert len({model.labels_[i] for i in range(4)}) == 1  # one shared cluster
        assert model.labels_[4] == model.labels_[0]  # border, pulled into that cluster
        assert model.labels_[5] == -1

    def test_core_points_helper_counts_self(self) -> None:
        X = np.array([[0.0], [0.1], [10.0]])
        dist = euclidean_distance(X, X)
        is_core, neighborhoods = _core_points(dist, eps=0.5, min_samples=2)
        np.testing.assert_array_equal(is_core, [True, True, False])
        np.testing.assert_array_equal(neighborhoods[2], [2])  # only itself


class TestIndependentReference:
    def test_matches_brute_force_reachability(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.normal(0, 0.2, size=(15, 2)),
                rng.normal([10, 10], 0.2, size=(15, 2)),
                np.array([[50.0, 50.0], [-50.0, -50.0]]),  # far, isolated noise
            ]
        )
        eps, min_samples = 0.6, 4

        model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        ref_clusters, ref_noise = _brute_force_partition(X, eps, min_samples)

        fitted_noise = set(np.where(model.labels_ == -1)[0].tolist())
        assert fitted_noise == ref_noise

        fitted_clusters = [
            set(np.where(model.labels_ == k)[0].tolist())
            for k in sorted(set(model.labels_.tolist()) - {-1})
        ]
        assert sorted(fitted_clusters, key=min) == sorted(ref_clusters, key=min)


class TestCorrectnessInvariants:
    def test_core_points_never_noise(self) -> None:
        X, _ = make_blobs(n_samples=80, centers=3, cluster_std=1.0, random_state=0)
        model = DBSCAN(eps=1.0, min_samples=4).fit(X)
        assert np.all(model.labels_[model.core_sample_indices_] != -1)

    def test_core_sample_indices_matches_core_count(self) -> None:
        X, _ = make_blobs(n_samples=80, centers=3, cluster_std=1.0, random_state=0)
        model = DBSCAN(eps=1.0, min_samples=4).fit(X)
        dist = euclidean_distance(X, X)
        expected_core = np.sum(dist <= 1.0, axis=1) >= 4  # noqa: PLR2004
        np.testing.assert_array_equal(
            model.core_sample_indices_, np.where(expected_core)[0]
        )

    def test_every_cluster_has_at_least_one_core_point(self) -> None:
        X, _ = make_blobs(n_samples=80, centers=3, cluster_std=1.0, random_state=0)
        model = DBSCAN(eps=1.0, min_samples=4).fit(X)
        core_labels = set(model.labels_[model.core_sample_indices_].tolist())
        for cluster_label in set(model.labels_.tolist()) - {-1}:
            assert cluster_label in core_labels

    def test_components_matches_x_at_core_indices(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=2, random_state=0)
        model = DBSCAN(eps=1.0, min_samples=3).fit(X)
        np.testing.assert_allclose(model.components_, X[model.core_sample_indices_])


class TestMetric:
    def test_euclidean_and_manhattan_can_disagree(self) -> None:
        # a point diagonally offset is within eps under one metric but not
        # the other -- constructed so the two metrics give different labels.
        X = np.array([[0.0, 0.0], [0.6, 0.6], [1.2, 0.0], [1.2, 1.2]])
        euclidean = DBSCAN(eps=0.9, min_samples=2, metric="euclidean").fit(X)
        manhattan = DBSCAN(eps=0.9, min_samples=2, metric="manhattan").fit(X)
        assert not np.array_equal(euclidean.labels_, manhattan.labels_)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="metric must be one of"):
            DBSCAN(metric="cosine").fit(np.zeros((3, 2)))


class TestContract:
    def test_fit_returns_self(self) -> None:
        X, _ = make_blobs(n_samples=20, centers=2, random_state=0)
        model = DBSCAN()
        assert model.fit(X) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, _ = make_blobs(n_samples=20, centers=2, random_state=0)
        model = DBSCAN(eps=0.8, min_samples=3, metric="manhattan")
        model.fit(X)
        assert model.get_params() == {
            "eps": 0.8,
            "min_samples": 3,
            "metric": "manhattan",
        }

    def test_non_positive_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps must be > 0"):
            DBSCAN(eps=0.0).fit(np.zeros((3, 2)))

    def test_bad_min_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="min_samples must be >= 1"):
            DBSCAN(min_samples=0).fit(np.zeros((3, 2)))

    def test_fit_predict_matches_fit_then_labels(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        a = DBSCAN(eps=1.5, min_samples=3).fit(X).labels_
        b = DBSCAN(eps=1.5, min_samples=3).fit_predict(X)
        np.testing.assert_array_equal(a, b)

    def test_no_predict_method(self) -> None:
        X, _ = make_blobs(n_samples=20, centers=2, random_state=0)
        model = DBSCAN().fit(X)
        assert not hasattr(model, "predict")

    def test_repr_round_trips(self) -> None:
        assert repr(DBSCAN()) == "DBSCAN(eps=0.5, min_samples=5, metric='euclidean')"


class TestBehavioral:
    def test_recovers_non_convex_moons(self) -> None:
        X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
        model = DBSCAN(eps=0.2, min_samples=5).fit(X)
        assert set(model.labels_.tolist()) - {-1} == {0, 1}
        assert _best_permutation_agreement(y, model.labels_, 2) > 0.95

    def test_detects_injected_outliers_as_noise(self) -> None:
        rng = np.random.default_rng(0)
        X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=0)
        outliers = rng.uniform(-20, 20, size=(10, 2))
        X_with_noise = np.vstack([X, outliers])
        model = DBSCAN(eps=0.8, min_samples=5).fit(X_with_noise)
        assert np.all(model.labels_[-10:] == -1)


class TestEdgeCases:
    def test_all_identical_points_form_one_cluster(self) -> None:
        X = np.zeros((10, 2))
        model = DBSCAN(eps=0.1, min_samples=5).fit(X)
        assert set(model.labels_.tolist()) == {0}
        assert len(model.core_sample_indices_) == 10  # noqa: PLR2004

    def test_all_far_apart_with_min_samples_one_are_singleton_clusters(self) -> None:
        X = np.array([[0.0], [100.0], [200.0]])
        model = DBSCAN(eps=0.5, min_samples=1).fit(X)
        assert set(model.labels_.tolist()) == {0, 1, 2}
        np.testing.assert_array_equal(model.core_sample_indices_, [0, 1, 2])

    def test_all_far_apart_with_min_samples_above_one_are_all_noise(self) -> None:
        X = np.array([[0.0], [100.0], [200.0]])
        model = DBSCAN(eps=0.5, min_samples=2).fit(X)
        assert set(model.labels_.tolist()) == {-1}
        assert model.core_sample_indices_.shape == (0,)

    def test_single_feature(self) -> None:
        X, _ = make_blobs(n_samples=30, n_features=1, centers=2, random_state=0)
        model = DBSCAN(eps=1.0, min_samples=3).fit(X)
        assert model.labels_.shape == (30,)


class TestHelpers:
    def test_expand_clusters_matches_fit(self) -> None:
        """The module-level helpers alone reproduce fit()'s labels_."""
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        dist = euclidean_distance(X, X)
        is_core, neighborhoods = _core_points(dist, eps=1.0, min_samples=3)
        labels = _expand_clusters(is_core, neighborhoods)
        model = DBSCAN(eps=1.0, min_samples=3).fit(X)
        np.testing.assert_array_equal(labels, model.labels_)
