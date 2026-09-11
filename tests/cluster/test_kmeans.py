"""Tests for scratchgrad.cluster.KMeans.

Tiers (plan.md section 3): K-means has an objective (within-cluster SSE)
but no gradient, so the correctness analogues are (a) the objective is
non-increasing across Lloyd iterations — both steps are exact argmins of
their subproblem (docs/derivations/kmeans.md section 4) — and (b) an
independent brute-force search over partitions of a tiny dataset agrees
with the fitted result. Plus init-strategy behaviour, empty-cluster
handling, `n_init`, convergence, determinism, the fit/predict contract,
behavioral recovery of a known partition, and edge cases.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scratchgrad.cluster import KMeans
from scratchgrad.cluster.kmeans import (
    _assign,
    _inertia,
    _init_kmeans_plusplus,
    _init_random,
    _lloyd,
    _update_centroids,
)
from scratchgrad.datasets import make_blobs
from scratchgrad.exceptions import NotFittedError


def _brute_force_min_inertia(X: np.ndarray, n_clusters: int) -> float:
    """Global-minimum WCSS over every possible labelling of X into n_clusters groups."""
    n = X.shape[0]
    best = np.inf
    for labels in itertools.product(range(n_clusters), repeat=n):
        labels = np.array(labels)
        inertia = 0.0
        for k in range(n_clusters):
            members = X[labels == k]
            if members.shape[0] > 0:
                inertia += np.sum((members - members.mean(axis=0)) ** 2)
        best = min(best, inertia)
    return best


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
    def test_assign_to_nearest_centroid(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 0.0], [9.0, 9.0], [10.0, 9.0]])
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        labels, dist = _assign(X, centers)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        assert dist.shape == (4, 2)

    def test_update_centroids_is_the_per_cluster_mean(self) -> None:
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
        labels = np.array([0, 0, 1, 1])
        _, dist = _assign(X, np.array([[0.0, 0.0], [10.0, 0.0]]))
        centers = _update_centroids(X, labels, dist, 2)
        np.testing.assert_allclose(centers, [[1.0, 0.0], [11.0, 0.0]])

    def test_explicit_init_converges_to_hand_computed_result(self) -> None:
        X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 0.0], [10.0, 2.0]])
        init = np.array([[0.0, 0.0], [10.0, 0.0]])
        model = KMeans(n_clusters=2, init=init, n_init=1).fit(X)
        np.testing.assert_allclose(
            sorted(model.cluster_centers_.tolist()), [[0.0, 1.0], [10.0, 1.0]]
        )
        assert model.inertia_ == pytest.approx(4.0)  # 2 clusters x 2 points x 1.0^2

    def test_inertia_helper_matches_manual_sum(self) -> None:
        X = np.array([[0.0], [3.0], [4.0]])
        centers = np.array([[0.0], [4.0]])
        labels, dist = _assign(X, centers)
        # 0 -> center 0 (d=0), 3 -> nearer to 4 (d=1), 4 -> center 1 (d=0)
        np.testing.assert_array_equal(labels, [0, 1, 1])
        assert _inertia(dist, labels) == pytest.approx(0.0 + 1.0 + 0.0)


class TestMonotonicity:
    def test_inertia_never_increases_across_lloyd_iterations(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 3))
        centers = _init_random(X, 4, rng)
        inertias = []
        for _ in range(15):
            labels, dist = _assign(X, centers)
            inertias.append(_inertia(dist, labels))
            centers = _update_centroids(X, labels, dist, 4)
        assert all(a >= b - 1e-9 for a, b in itertools.pairwise(inertias))

    def test_lloyd_final_inertia_leq_initial_assignment_inertia(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(80, 2))
        centers0 = _init_kmeans_plusplus(X, 3, rng)
        labels0, dist0 = _assign(X, centers0)
        start = _inertia(dist0, labels0)
        _, _, end, _ = _lloyd(X, centers0, 3, max_iter=100, tol=0.0)
        assert end <= start + 1e-9


class TestIndependentReference:
    def test_matches_brute_force_global_minimum(self) -> None:
        X = np.array(
            [[0.0, 0.0], [0.1, 0.2], [-0.1, 0.1], [0.2, -0.1], [9.0, 9.0], [9.1, 8.9]]
        )
        best_possible = _brute_force_min_inertia(X, 2)
        model = KMeans(n_clusters=2, n_init=20, random_state=0).fit(X)
        assert model.inertia_ == pytest.approx(best_possible, abs=1e-6)


class TestInitStrategies:
    def test_kmeans_plusplus_more_reliable_than_random_on_uneven_data(self) -> None:
        # one dense 100-point cluster plus 3 far, well-separated singleton
        # points -- random init is much more likely to waste two seeds
        # inside the dense blob than k-means++'s D(x)^2-weighted draw,
        # which is heavily biased toward the far-away singletons.
        rng = np.random.default_rng(0)
        dense = rng.normal(0.0, 0.3, size=(100, 2))
        outliers = np.array([[50.0, 50.0], [-50.0, 50.0], [50.0, -50.0]])
        X = np.vstack([dense, outliers])

        def final_inertia(init: str, seed: int) -> float:
            return (
                KMeans(n_clusters=4, init=init, n_init=1, random_state=seed)
                .fit(X)
                .inertia_
            )

        best_partition_inertia = (
            KMeans(n_clusters=4, n_init=50, random_state=0).fit(X).inertia_
        )
        n_seeds = 30
        plusplus_hits = sum(
            final_inertia("k-means++", s)
            == pytest.approx(best_partition_inertia, abs=1e-6)
            for s in range(n_seeds)
        )
        random_hits = sum(
            final_inertia("random", s)
            == pytest.approx(best_partition_inertia, abs=1e-6)
            for s in range(n_seeds)
        )
        assert plusplus_hits > random_hits


class TestEmptyCluster:
    def test_no_crash_and_uses_all_k_labels(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack(
            [rng.normal(0, 0.1, size=(20, 2)), rng.normal([20, 20], 0.1, size=(20, 2))]
        )
        # third seed is nowhere near any data -> its cluster starts empty
        init = np.array([[0.0, 0.0], [20.0, 20.0], [1000.0, 1000.0]])
        model = KMeans(n_clusters=3, init=init, n_init=1, random_state=0).fit(X)
        assert set(model.labels_.tolist()) == {0, 1, 2}

    def test_stolen_point_excluded_from_its_original_clusters_mean(self) -> None:
        # regression test: a stolen point must not count toward both its
        # original cluster's mean AND be the new singleton centroid.
        X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 0, 0])  # cluster 1 starts empty
        _, dist = _assign(X, np.array([[0.0, 0.0], [-5.0, -5.0]]))
        centers = _update_centroids(X, labels, dist, 2)
        # farthest point from centroid 0 is [10, 10] -> stolen into cluster 1
        np.testing.assert_allclose(centers[1], [10.0, 10.0])
        # cluster 0's mean must be over the two [0, 0] points ONLY
        np.testing.assert_allclose(centers[0], [0.0, 0.0])


class TestNInit:
    def test_best_of_n_init_is_leq_the_first_restart_alone(self) -> None:
        # n_init=1 and n_init=8 draw from the same rng given the same seed,
        # so n_init=1's result IS one of n_init=8's candidates -- the best
        # of 8 can only match or beat it, never do worse.
        X, _ = make_blobs(n_samples=100, centers=4, cluster_std=3.0, random_state=0)
        single = KMeans(n_clusters=4, n_init=1, random_state=0).fit(X).inertia_
        best_of_many = KMeans(n_clusters=4, n_init=8, random_state=0).fit(X).inertia_
        assert best_of_many <= single + 1e-9

    def test_explicit_init_forces_n_init_to_one(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=2, random_state=0)
        init = X[[0, 20]].copy()
        # a huge n_init would be wasted repeating an identical deterministic
        # run -- confirm it doesn't change the (fast) outcome.
        model = KMeans(n_clusters=2, init=init, n_init=1000).fit(X)
        again = KMeans(n_clusters=2, init=init, n_init=1).fit(X)
        np.testing.assert_allclose(model.cluster_centers_, again.cluster_centers_)


class TestConvergence:
    def test_n_iter_never_exceeds_max_iter(self) -> None:
        X, _ = make_blobs(n_samples=200, centers=5, random_state=0)
        model = KMeans(n_clusters=5, max_iter=3, random_state=0).fit(X)
        assert model.n_iter_ <= 3  # noqa: PLR2004

    def test_max_iter_one_still_returns_a_valid_result(self) -> None:
        X, _ = make_blobs(n_samples=50, centers=3, random_state=0)
        model = KMeans(n_clusters=3, max_iter=1, n_init=1, random_state=0).fit(X)
        assert model.n_iter_ == 1
        assert model.labels_.shape == (50,)


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        X, _ = make_blobs(n_samples=150, centers=4, random_state=0)
        a = KMeans(n_clusters=4, random_state=42).fit(X)
        b = KMeans(n_clusters=4, random_state=42).fit(X)
        np.testing.assert_allclose(a.cluster_centers_, b.cluster_centers_)
        np.testing.assert_array_equal(a.labels_, b.labels_)

    def test_different_seed_can_differ(self) -> None:
        X, _ = make_blobs(n_samples=150, centers=6, cluster_std=5.0, random_state=0)
        a = KMeans(n_clusters=6, n_init=1, random_state=1).fit(X)
        b = KMeans(n_clusters=6, n_init=1, random_state=2).fit(X)
        a_centers = sorted(a.cluster_centers_.tolist())
        b_centers = sorted(b.cluster_centers_.tolist())
        assert not np.allclose(a_centers, b_centers)

    def test_explicit_init_is_fully_deterministic(self) -> None:
        X, _ = make_blobs(n_samples=60, centers=3, random_state=0)
        init = X[[0, 20, 40]].copy()
        a = KMeans(n_clusters=3, init=init).fit(X)
        b = KMeans(n_clusters=3, init=init).fit(X)
        np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)


class TestContract:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            KMeans().predict(np.zeros((2, 2)))

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            KMeans().transform(np.zeros((2, 2)))

    def test_fit_returns_self(self) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        model = KMeans(n_clusters=2)
        assert model.fit(X) is model

    def test_fit_does_not_mutate_hyperparameters(self) -> None:
        X, _ = make_blobs(n_samples=30, centers=2, random_state=0)
        model = KMeans(n_clusters=3, n_init=5, max_iter=50, tol=1e-3, random_state=7)
        model.fit(X)
        assert model.get_params() == {
            "n_clusters": 3,
            "init": "k-means++",
            "n_init": 5,
            "max_iter": 50,
            "tol": 1e-3,
            "random_state": 7,
        }

    def test_predict_rejects_wrong_feature_count(self) -> None:
        X, _ = make_blobs(n_samples=30, n_features=3, centers=2, random_state=0)
        model = KMeans(n_clusters=2, random_state=0).fit(X)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((4, 2)))

    def test_transform_rejects_wrong_feature_count(self) -> None:
        X, _ = make_blobs(n_samples=30, n_features=3, centers=2, random_state=0)
        model = KMeans(n_clusters=2, random_state=0).fit(X)
        with pytest.raises(ValueError, match="features"):
            model.transform(np.zeros((4, 2)))

    def test_bad_n_clusters_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            KMeans(n_clusters=0).fit(X)

    def test_n_clusters_larger_than_n_samples_raises(self) -> None:
        X, _ = make_blobs(n_samples=5, centers=2, random_state=0)
        with pytest.raises(ValueError, match="larger than the number of"):
            KMeans(n_clusters=6).fit(X)

    def test_bad_init_string_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="init must be one of"):
            KMeans(n_clusters=2, init="forgy").fit(X)

    def test_bad_init_array_shape_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, n_features=2, centers=2, random_state=0)
        with pytest.raises(ValueError, match="init array must have shape"):
            KMeans(n_clusters=2, init=np.zeros((3, 2))).fit(X)

    def test_bad_n_init_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="n_init must be >= 1"):
            KMeans(n_clusters=2, n_init=0).fit(X)

    def test_bad_max_iter_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="max_iter must be >= 1"):
            KMeans(n_clusters=2, max_iter=0).fit(X)

    def test_negative_tol_raises(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=2, random_state=0)
        with pytest.raises(ValueError, match="tol must be >= 0"):
            KMeans(n_clusters=2, tol=-1.0).fit(X)

    def test_score_is_negative_inertia(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        model = KMeans(n_clusters=3, random_state=0).fit(X)
        assert model.score(X) == pytest.approx(-model.inertia_)

    def test_fit_predict_matches_fit_then_labels(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        a = KMeans(n_clusters=3, random_state=0).fit(X).labels_
        b = KMeans(n_clusters=3, random_state=0).fit_predict(X)
        np.testing.assert_array_equal(a, b)

    def test_fit_transform_matches_fit_then_transform(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        a = KMeans(n_clusters=3, random_state=0).fit(X).transform(X)
        b = KMeans(n_clusters=3, random_state=0).fit_transform(X)
        np.testing.assert_allclose(a, b)

    def test_repr_round_trips(self) -> None:
        assert repr(KMeans()) == (
            "KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, "
            "tol=0.0001, random_state=None)"
        )


class TestBehavioral:
    def test_recovers_well_separated_blobs(self) -> None:
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=0)
        model = KMeans(n_clusters=4, random_state=0).fit(X)
        assert _best_permutation_agreement(y, model.labels_, 4) > 0.95

    def test_transform_distances_agree_with_predict(self) -> None:
        X, _ = make_blobs(n_samples=100, centers=3, random_state=0)
        model = KMeans(n_clusters=3, random_state=0).fit(X)
        dist = model.transform(X)
        np.testing.assert_array_equal(model.predict(X), np.argmin(dist, axis=1))


class TestEdgeCases:
    def test_single_feature(self) -> None:
        X, _ = make_blobs(n_samples=40, n_features=1, centers=2, random_state=0)
        model = KMeans(n_clusters=2, random_state=0).fit(X)
        assert model.cluster_centers_.shape == (2, 1)

    def test_n_clusters_equals_n_samples_gives_zero_inertia(self) -> None:
        X, _ = make_blobs(n_samples=10, centers=3, random_state=0)
        model = KMeans(n_clusters=10, n_init=1, random_state=0).fit(X)
        assert model.inertia_ == pytest.approx(0.0, abs=1e-9)

    def test_n_clusters_one_equals_the_global_mean(self) -> None:
        X, _ = make_blobs(n_samples=40, centers=3, random_state=0)
        model = KMeans(n_clusters=1, random_state=0).fit(X)
        np.testing.assert_allclose(model.cluster_centers_[0], X.mean(axis=0))
        assert model.inertia_ == pytest.approx(np.sum((X - X.mean(axis=0)) ** 2))
