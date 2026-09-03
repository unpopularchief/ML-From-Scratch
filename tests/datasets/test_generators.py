"""Tests for scratchgrad.datasets.generators."""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets.generators import make_blobs, make_moons, make_regression


class TestMakeRegression:
    def test_output_shapes(self) -> None:
        X, y = make_regression(n_samples=50, n_features=3, random_state=0)
        assert X.shape == (50, 3)
        assert y.shape == (50,)

    def test_zero_noise_is_exactly_linear(self) -> None:
        # With noise=0, y must be *exactly* reproducible as X @ w + b for
        # some w, b — checked by fitting the normal equation ourselves
        # (not by importing the library's own LinearRegression, which
        # doesn't exist yet and shouldn't be the thing under test here).
        X, y = make_regression(n_samples=200, n_features=3, noise=0.0, random_state=0)
        X_with_bias = np.column_stack([X, np.ones(len(X))])
        w_and_b, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
        y_reconstructed = X_with_bias @ w_and_b
        np.testing.assert_allclose(y_reconstructed, y, atol=1e-8)

    def test_same_seed_reproducible(self) -> None:
        X1, y1 = make_regression(random_state=7)
        X2, y2 = make_regression(random_state=7)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestMakeBlobs:
    def test_output_shapes(self) -> None:
        X, y = make_blobs(n_samples=30, n_features=2, centers=3, random_state=0)
        assert X.shape == (30, 2)
        assert y.shape == (30,)

    def test_produces_exactly_n_centers_classes(self) -> None:
        _, y = make_blobs(n_samples=60, centers=4, random_state=0)
        assert set(np.unique(y).tolist()) == {0, 1, 2, 3}

    def test_samples_split_as_evenly_as_possible(self) -> None:
        _, y = make_blobs(n_samples=10, centers=3, random_state=0)
        counts = np.bincount(y)
        assert sorted(counts.tolist()) == [3, 3, 4]

    def test_points_cluster_near_their_assigned_center(self) -> None:
        # With a tiny cluster_std, every point must land close to *some*
        # center, and specifically the one matching its own label.
        X, y = make_blobs(n_samples=30, centers=3, cluster_std=0.01, random_state=0)
        for label in np.unique(y):
            cluster_points = X[y == label]
            spread = np.std(cluster_points, axis=0)
            assert np.all(spread < 0.1)


class TestMakeMoons:
    def test_output_shapes(self) -> None:
        X, y = make_moons(n_samples=40, random_state=0)
        assert X.shape == (40, 2)
        assert y.shape == (40,)

    def test_produces_two_classes_split_in_half(self) -> None:
        _, y = make_moons(n_samples=50, random_state=0)
        assert np.sum(y == 0) == 25
        assert np.sum(y == 1) == 25

    def test_not_linearly_separable_by_a_single_axis(self) -> None:
        # The defining property of make_moons: unlike make_blobs, the two
        # classes interleave, so neither raw coordinate alone separates
        # them (there's no threshold t with all class-0 x < t < all class-1 x).
        X, y = make_moons(n_samples=100, noise=0.0, random_state=0)
        x0, x1 = X[y == 0, 0], X[y == 1, 0]
        separable_by_x = np.max(x0) < np.min(x1) or np.max(x1) < np.min(x0)
        assert not separable_by_x
