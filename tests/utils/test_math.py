"""Tests for scratchgrad.utils.math."""

from __future__ import annotations

import numpy as np

from scratchgrad.utils.math import logsumexp, sigmoid, softmax


class TestSigmoid:
    def test_zero_maps_to_half(self) -> None:
        assert sigmoid(np.array([0.0]))[0] == 0.5

    def test_matches_naive_formula_near_zero(self) -> None:
        # Away from the overflow-prone extremes, the branch-free textbook
        # formula 1 / (1 + e^-z) is itself a trustworthy reference.
        z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        expected = 1.0 / (1.0 + np.exp(-z))
        np.testing.assert_allclose(sigmoid(z), expected, atol=1e-12)

    def test_no_overflow_for_large_magnitude_input(self) -> None:
        z = np.array([-1000.0, 1000.0])
        result = sigmoid(z)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-12)

    def test_output_bounded_in_closed_unit_interval(self) -> None:
        # Mathematically sigmoid's range is the *open* interval (0, 1), but
        # float64 can't represent 1 - 8e-17: for |z| gtrsim 37 the computed
        # result saturates to exactly 0.0 or 1.0. That's an expected
        # floating-point fact, not a bug, so the wide-range check only
        # asserts the closed interval.
        z = np.linspace(-50, 50, 101)
        result = sigmoid(z)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_output_strictly_between_zero_and_one_for_moderate_input(self) -> None:
        # Within a range that doesn't hit float64's saturation point, the
        # open-interval property does hold and is checked strictly.
        z = np.linspace(-30, 30, 101)
        result = sigmoid(z)
        assert np.all(result > 0.0) and np.all(result < 1.0)

    def test_symmetry_sigmoid_neg_z_equals_one_minus_sigmoid_z(self) -> None:
        z = np.array([-3.0, -1.0, 0.5, 4.0])
        np.testing.assert_allclose(sigmoid(-z), 1.0 - sigmoid(z), atol=1e-12)


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        z = np.array([1.0, 2.0, 3.0])
        assert np.isclose(np.sum(softmax(z)), 1.0)

    def test_matches_naive_formula_near_zero(self) -> None:
        z = np.array([0.1, -0.3, 0.7])
        expected = np.exp(z) / np.sum(np.exp(z))
        np.testing.assert_allclose(softmax(z), expected, atol=1e-12)

    def test_no_overflow_for_large_input(self) -> None:
        z = np.array([1000.0, 1001.0, 999.0])
        result = softmax(z)
        assert np.all(np.isfinite(result))
        assert np.isclose(np.sum(result), 1.0)

    def test_uniform_input_gives_uniform_output(self) -> None:
        z = np.full(5, 3.0)
        np.testing.assert_allclose(softmax(z), np.full(5, 0.2), atol=1e-12)

    def test_operates_along_given_axis_for_batched_input(self) -> None:
        z = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        result = softmax(z, axis=1)
        np.testing.assert_allclose(np.sum(result, axis=1), [1.0, 1.0])


class TestLogSumExp:
    def test_matches_naive_formula_near_zero(self) -> None:
        z = np.array([0.1, -0.3, 0.7])
        expected = np.log(np.sum(np.exp(z)))
        np.testing.assert_allclose(logsumexp(z), expected, atol=1e-12)

    def test_no_overflow_for_large_input(self) -> None:
        z = np.array([1000.0, 1001.0])
        assert np.isfinite(logsumexp(z))

    def test_lower_bound_is_max(self) -> None:
        # logsumexp(z) >= max(z), since it equals max(z) + log(sum of
        # terms each <= 1, at least one of which is exactly 1).
        z = np.array([1.0, 5.0, 2.0])
        assert logsumexp(z) >= np.max(z)

    def test_relates_to_softmax_via_its_own_gradient(self) -> None:
        # d/dz_i logsumexp(z) = softmax(z)_i - this is the identity that
        # makes logsumexp the right building block for cross-entropy loss
        # later. Checked here with finite differences against the actual
        # analytic quantity (softmax), not just asserted.
        from tests.helpers.gradcheck import numerical_gradient

        z = np.array([0.5, -1.0, 2.0])
        numerical_grad = numerical_gradient(lambda v: float(logsumexp(v)), z)
        np.testing.assert_allclose(numerical_grad, softmax(z), atol=1e-6)
