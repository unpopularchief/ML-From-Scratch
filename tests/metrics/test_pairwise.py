"""Tests for scratchgrad.metrics.pairwise — all analytic (tier 1) checks."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.metrics.pairwise import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)


class TestEuclideanDistance:
    def test_hand_computed_3_4_5_triangle(self) -> None:
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        assert euclidean_distance(a, b)[0, 0] == pytest.approx(5.0)

    def test_zero_distance_to_self(self) -> None:
        a = np.array([[1.0, 2.0, 3.0]])
        assert euclidean_distance(a, a)[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_output_shape_is_n_a_by_n_b(self) -> None:
        a = np.zeros((3, 2))
        b = np.zeros((5, 2))
        assert euclidean_distance(a, b).shape == (3, 5)

    def test_symmetric(self) -> None:
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(euclidean_distance(a, b), euclidean_distance(b, a).T)

    def test_matches_manhattan_in_one_dimension(self) -> None:
        # In 1D, |x - y| = sqrt((x - y)^2), so Euclidean and Manhattan
        # distance coincide exactly.
        a = np.array([[1.0], [5.0]])
        b = np.array([[3.0]])
        np.testing.assert_allclose(euclidean_distance(a, b), manhattan_distance(a, b))


class TestManhattanDistance:
    def test_hand_computed_example(self) -> None:
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        # |3-0| + |4-0| = 7
        assert manhattan_distance(a, b)[0, 0] == pytest.approx(7.0)

    def test_zero_distance_to_self(self) -> None:
        a = np.array([[1.0, 2.0, 3.0]])
        assert manhattan_distance(a, a)[0, 0] == 0.0


class TestCosineSimilarity:
    def test_identical_direction_scores_one(self) -> None:
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[2.0, 4.0, 6.0]])  # same direction, different magnitude
        assert cosine_similarity(a, b)[0, 0] == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert cosine_similarity(a, b)[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_opposite_direction_scores_negative_one(self) -> None:
        a = np.array([[1.0, 2.0]])
        b = np.array([[-1.0, -2.0]])
        assert cosine_similarity(a, b)[0, 0] == pytest.approx(-1.0)

    def test_raises_for_zero_vector(self) -> None:
        a = np.array([[0.0, 0.0]])
        b = np.array([[1.0, 1.0]])
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity(a, b)
