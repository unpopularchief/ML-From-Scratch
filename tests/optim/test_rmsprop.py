"""Tests for scratchgrad.optim.RMSprop.

Tiers (plan.md section 3, docs/derivations/optim.md section 8): a
hand-computed analytic step, a convergence check, an ill-conditioned
comparison against plain SGD, and the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.optim import SGD, RMSprop


class TestContract:
    def test_non_positive_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            RMSprop(lr=0.0)

    @pytest.mark.parametrize("beta", [0.0, 1.0, -0.1])
    def test_beta_out_of_range_raises(self, beta) -> None:
        with pytest.raises(ValueError, match="beta"):
            RMSprop(beta=beta)

    def test_non_positive_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            RMSprop(eps=0.0)


class TestAnalytic:
    def test_single_step_matches_hand_computation(self) -> None:
        # s0 = 0. s1 = 0.9*0 + 0.1*g^2 = 0.1*4 = 0.4.
        # theta1 = 1 - lr*g/(sqrt(0.4)+eps).
        lr, beta, eps = 0.1, 0.9, 1e-8
        opt = RMSprop(lr=lr, beta=beta, eps=eps)
        theta = np.array([1.0])
        g = np.array([2.0])

        opt.step([theta], [g])

        expected = 1.0 - lr * 2.0 / (np.sqrt(0.4) + eps)
        np.testing.assert_allclose(theta, [expected])


class TestBehavioral:
    def test_converges_on_a_quadratic_bowl(self) -> None:
        # Dividing g_t by its own running magnitude sqrt(s_t) makes each
        # RMSprop step roughly constant size ~lr, independent of how large
        # or small g_t itself is -- unlike a plain gradient step, which
        # shrinks automatically as theta nears the optimum. So progress is
        # steady (~lr per step, not accelerating like SGD would near a
        # bowl) rather than fast, and once close, theta settles into a
        # residual "noise ball" whose radius scales with lr (measured:
        # ~0.7*lr) instead of shrinking further with more steps. A small lr
        # and enough steps to cover the initial distance are both needed.
        A = np.diag([1.0, 4.0])
        theta = np.array([5.0, -3.0])
        opt = RMSprop(lr=0.01)

        for _ in range(2000):
            grad = A @ theta
            opt.step([theta], [grad])

        assert np.linalg.norm(theta) < 1e-2

    def test_beats_plain_sgd_on_an_ill_conditioned_quadratic(self) -> None:
        # RMSprop divides each coordinate's step by its own gradient
        # magnitude, so one global lr works for both axes even though they
        # have very different curvature -- unlike SGD, which must use a
        # small lr to stay stable on the steep axis.
        A = np.diag([1.0, 100.0])
        theta_sgd = np.array([10.0, 1.0])
        theta_rms = theta_sgd.copy()
        sgd = SGD(lr=0.005)
        rmsprop = RMSprop(lr=0.1)

        for _ in range(100):
            sgd.step([theta_sgd], [A @ theta_sgd])
            rmsprop.step([theta_rms], [A @ theta_rms])

        loss_sgd = 0.5 * theta_sgd @ A @ theta_sgd
        loss_rms = 0.5 * theta_rms @ A @ theta_rms
        assert loss_rms < loss_sgd


class TestDeterminism:
    def test_state_shape_matches_params(self) -> None:
        opt = RMSprop()
        opt.step([np.zeros((2, 3))], [np.ones((2, 3))])
        assert opt.square_avg_[0].shape == (2, 3)
