"""Tests for scratchgrad.optim.Adam.

Tiers (plan.md section 3, docs/derivations/optim.md section 8): a
hand-computed analytic first step (bias correction dominates here, since
t=1 is where it matters most), a convergence check, an ill-conditioned
comparison against plain SGD, and the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.optim import SGD, Adam


class TestContract:
    def test_non_positive_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            Adam(lr=0.0)

    @pytest.mark.parametrize("beta1", [0.0, 1.0])
    def test_beta1_out_of_range_raises(self, beta1) -> None:
        with pytest.raises(ValueError, match="beta1"):
            Adam(beta1=beta1)

    @pytest.mark.parametrize("beta2", [0.0, 1.0])
    def test_beta2_out_of_range_raises(self, beta2) -> None:
        with pytest.raises(ValueError, match="beta2"):
            Adam(beta2=beta2)

    def test_non_positive_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            Adam(eps=0.0)


class TestAnalytic:
    def test_first_step_matches_hand_computation(self) -> None:
        # m0=s0=0. m1 = 0.1*g = 0.1*2 = 0.2. s1 = 0.001*g^2 = 0.001*4 = 0.004.
        # bias1 = 1 - 0.9^1 = 0.1, bias2 = 1 - 0.999^1 = 0.001.
        # m_hat = 0.2/0.1 = 2.0 = g (bias correction exactly recovers g at
        # t=1, since m1 = (1-beta1)*g and bias1 = 1-beta1).
        # s_hat = 0.004/0.001 = 4.0 = g^2, same reasoning.
        lr, beta1, beta2, eps = 0.1, 0.9, 0.999, 1e-8
        opt = Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        theta = np.array([1.0])
        g = np.array([2.0])

        opt.step([theta], [g])

        expected = 1.0 - lr * 2.0 / (np.sqrt(4.0) + eps)
        np.testing.assert_allclose(theta, [expected])

    def test_bias_correction_recovers_the_gradient_exactly_at_t_equals_1(self) -> None:
        # A direct check of the general claim used above: for ANY g, at
        # t=1, m_hat == g and s_hat == g**2 exactly (up to eps in the
        # denominator), independent of beta1/beta2.
        eps = 1e-12
        opt = Adam(lr=0.01, beta1=0.5, beta2=0.8, eps=eps)
        theta = np.array([0.0])
        g = np.array([3.7])

        opt.step([theta], [g])

        expected = 0.0 - 0.01 * g / (np.sqrt(g**2) + eps)
        np.testing.assert_allclose(theta, expected)


class TestBehavioral:
    def test_converges_on_a_quadratic_bowl(self) -> None:
        A = np.diag([1.0, 4.0])
        theta = np.array([5.0, -3.0])
        opt = Adam(lr=0.1)

        for _ in range(500):
            grad = A @ theta
            opt.step([theta], [grad])

        assert np.linalg.norm(theta) < 1e-2

    def test_beats_plain_sgd_on_an_ill_conditioned_quadratic(self) -> None:
        A = np.diag([1.0, 100.0])
        theta_sgd = np.array([10.0, 1.0])
        theta_adam = theta_sgd.copy()
        sgd = SGD(lr=0.005)
        adam = Adam(lr=0.1)

        for _ in range(100):
            sgd.step([theta_sgd], [A @ theta_sgd])
            adam.step([theta_adam], [A @ theta_adam])

        loss_sgd = 0.5 * theta_sgd @ A @ theta_sgd
        loss_adam = 0.5 * theta_adam @ A @ theta_adam
        assert loss_adam < loss_sgd


class TestDeterminism:
    def test_state_shapes_and_step_counter(self) -> None:
        opt = Adam()
        opt.step([np.zeros((2, 3))], [np.ones((2, 3))])
        assert opt.m_[0].shape == (2, 3)
        assert opt.s_[0].shape == (2, 3)
        assert opt.t_ == 1
        opt.step([np.zeros((2, 3))], [np.ones((2, 3))])
        assert opt.t_ == 2
