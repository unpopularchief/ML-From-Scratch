"""Tests for the gradient-checking helper itself.

This is the one piece of test infrastructure the whole project leans on
from M3 onward, so it earns tests of its own: not just "does it pass for
a correct gradient" but "does it actually fail for a wrong one" — a
gradient checker that always passes would be worse than having none.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.helpers.gradcheck import gradient_check, numerical_gradient, relative_error


class TestNumericalGradient:
    def test_matches_known_gradient_of_quadratic(self) -> None:
        # f(x) = sum(x^2), grad f(x) = 2x — simple enough to verify by hand.
        x = np.array([1.0, -2.0, 3.0])
        grad = numerical_gradient(lambda v: float(np.sum(v**2)), x)
        np.testing.assert_allclose(grad, 2.0 * x, atol=1e-6)

    def test_matches_known_gradient_of_dot_product(self) -> None:
        # f(x) = a . x, grad f(x) = a
        a = np.array([2.0, -1.0, 0.5])
        grad = numerical_gradient(lambda v: float(a @ v), np.array([1.0, 1.0, 1.0]))
        np.testing.assert_allclose(grad, a, atol=1e-6)

    def test_does_not_mutate_input_array(self) -> None:
        # numerical_gradient perturbs x in place during the loop; it must
        # restore every coordinate afterward rather than leaking a
        # perturbed value back to the caller.
        x = np.array([1.0, 2.0, 3.0])
        original = x.copy()
        numerical_gradient(lambda v: float(np.sum(v**2)), x)
        np.testing.assert_array_equal(x, original)

    def test_handles_2d_input_shape(self) -> None:
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        grad = numerical_gradient(lambda v: float(np.sum(v**2)), x)
        np.testing.assert_allclose(grad, 2.0 * x, atol=1e-6)


class TestRelativeError:
    def test_zero_for_identical_gradients(self) -> None:
        g = np.array([1.0, -2.0, 0.5])
        assert relative_error(g, g) == 0.0

    def test_positive_for_differing_gradients(self) -> None:
        assert relative_error(np.array([1.0]), np.array([2.0])) > 0.0

    def test_floor_prevents_division_by_zero_for_all_zero_gradients(self) -> None:
        # Without eps_floor, comparing two all-zero gradients would divide
        # 0 by 0 (nan) instead of correctly reporting no disagreement.
        error = relative_error(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
        assert error == 0.0


class TestGradientCheck:
    def test_passes_for_a_correct_gradient(self) -> None:
        x = np.array([1.0, -2.0, 3.0])
        gradient_check(lambda v: float(np.sum(v**2)), analytic_grad=2.0 * x, x=x)

    def test_fails_for_a_wrong_gradient(self) -> None:
        x = np.array([1.0, -2.0, 3.0])
        wrong_grad = 3.0 * x  # correct gradient is 2*x
        with pytest.raises(AssertionError, match="Gradient check failed"):
            gradient_check(lambda v: float(np.sum(v**2)), analytic_grad=wrong_grad, x=x)

    def test_fails_for_a_sign_error(self) -> None:
        x = np.array([1.0, -2.0, 3.0])
        with pytest.raises(AssertionError):
            gradient_check(lambda v: float(np.sum(v**2)), analytic_grad=-2.0 * x, x=x)
