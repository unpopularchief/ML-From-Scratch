"""Tests for scratchgrad.optim.Momentum.

Tiers (plan.md section 3, docs/derivations/optim.md section 8): a
hand-computed analytic step, the momentum=0 reduction to plain SGD, a
convergence check, an ill-conditioned-quadratic race against plain SGD,
and the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.optim import SGD, Momentum


class TestContract:
    def test_non_positive_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            Momentum(lr=0.0)

    @pytest.mark.parametrize("momentum", [-0.1, 1.0, 1.5])
    def test_momentum_out_of_range_raises(self, momentum) -> None:
        with pytest.raises(ValueError, match="momentum"):
            Momentum(momentum=momentum)


class TestAnalytic:
    def test_two_steps_match_hand_computation(self) -> None:
        # v0 = 0. step1: v = -0.1*1 = -0.1, theta = 1 - 0.1 = 0.9.
        # step2: v = 0.9*(-0.1) - 0.1*1 = -0.19, theta = 0.9 - 0.19 = 0.71.
        opt = Momentum(lr=0.1, momentum=0.9)
        theta = np.array([1.0])
        opt.step([theta], [np.array([1.0])])
        opt.step([theta], [np.array([1.0])])
        np.testing.assert_allclose(theta, [0.71])

    def test_velocity_buffer_shape_matches_params(self) -> None:
        opt = Momentum()
        opt.step([np.zeros((2, 3))], [np.ones((2, 3))])
        assert opt.velocity_[0].shape == (2, 3)


class TestReduction:
    def test_zero_momentum_matches_plain_sgd(self) -> None:
        lr = 0.05
        mom = Momentum(lr=lr, momentum=0.0)
        sgd = SGD(lr=lr)
        theta_mom, theta_sgd = np.array([3.0, -1.0]), np.array([3.0, -1.0])

        rng = np.random.default_rng(0)
        for _ in range(20):
            grad = rng.standard_normal(2)
            mom.step([theta_mom], [grad.copy()])
            sgd.step([theta_sgd], [grad.copy()])

        np.testing.assert_array_equal(theta_mom, theta_sgd)


class TestBehavioral:
    def test_converges_on_a_quadratic_bowl(self) -> None:
        A = np.diag([1.0, 4.0])
        theta = np.array([5.0, -3.0])
        opt = Momentum(lr=0.1, momentum=0.9)

        for _ in range(200):
            grad = A @ theta
            opt.step([theta], [grad])

        assert np.linalg.norm(theta) < 1e-3

    def test_beats_plain_sgd_on_an_ill_conditioned_quadratic(self) -> None:
        # Condition number 100: SGD needs a small lr to stay stable along the
        # steep axis, which makes it slow along the shallow one. Momentum
        # accumulates speed along the shallow axis without needing a larger
        # global lr.
        A = np.diag([1.0, 100.0])
        lr = 0.005
        theta_sgd = np.array([10.0, 1.0])
        theta_mom = theta_sgd.copy()
        sgd = SGD(lr=lr)
        mom = Momentum(lr=lr, momentum=0.9)

        n_steps = 100
        for _ in range(n_steps):
            sgd.step([theta_sgd], [A @ theta_sgd])
            mom.step([theta_mom], [A @ theta_mom])

        loss_sgd = 0.5 * theta_sgd @ A @ theta_sgd
        loss_mom = 0.5 * theta_mom @ A @ theta_mom
        assert loss_mom < loss_sgd
