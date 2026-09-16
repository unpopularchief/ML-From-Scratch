"""Tests for scratchgrad.optim.SGD.

Tiers (plan.md section 3, docs/derivations/optim.md section 8): a
hand-computed analytic step, a convergence check on a quadratic bowl, the
contract (invalid hyperparameters raise), and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.optim import SGD


class TestContract:
    def test_non_positive_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            SGD(lr=0.0)
        with pytest.raises(ValueError, match="lr"):
            SGD(lr=-1.0)


class TestAnalytic:
    def test_single_step_matches_hand_computation(self) -> None:
        opt = SGD(lr=0.1)
        params = [np.array([1.0, 2.0]), np.array([[3.0]])]
        grads = [np.array([1.0, 1.0]), np.array([[2.0]])]

        opt.step(params, grads)

        np.testing.assert_allclose(params[0], [0.9, 1.9])
        np.testing.assert_allclose(params[1], [[2.8]])

    def test_mutates_params_in_place(self) -> None:
        opt = SGD(lr=0.5)
        theta = np.array([2.0])
        params = [theta]
        opt.step(params, [np.array([1.0])])
        assert theta is params[0]
        np.testing.assert_allclose(theta, [1.5])


class TestBehavioral:
    def test_converges_on_a_quadratic_bowl(self) -> None:
        # L(theta) = 1/2 theta^T A theta, grad = A theta, minimum at 0.
        A = np.diag([1.0, 4.0])
        theta = np.array([5.0, -3.0])
        opt = SGD(lr=0.1)

        for _ in range(200):
            grad = A @ theta
            opt.step([theta], [grad])

        assert np.linalg.norm(theta) < 1e-3


class TestDeterminism:
    def test_identical_step_sequences_are_byte_identical(self) -> None:
        opt_a, opt_b = SGD(lr=0.05), SGD(lr=0.05)
        theta_a, theta_b = np.array([1.0, -2.0]), np.array([1.0, -2.0])

        for _ in range(10):
            grad = np.array([0.3, -0.1])
            opt_a.step([theta_a], [grad.copy()])
            opt_b.step([theta_b], [grad.copy()])

        np.testing.assert_array_equal(theta_a, theta_b)
