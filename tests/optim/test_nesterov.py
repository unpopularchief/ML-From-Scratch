"""Tests for scratchgrad.optim.Nesterov.

Tiers (plan.md section 3, docs/derivations/optim.md section 8): the
substitution-correctness check against the literal lookahead-gradient
formula (the real correctness test for the section 4 reformulation), the
momentum=0 reduction to plain SGD, a convergence check, and the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.optim import SGD, Nesterov


class TestContract:
    def test_non_positive_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr"):
            Nesterov(lr=0.0)

    @pytest.mark.parametrize("momentum", [-0.1, 1.0])
    def test_momentum_out_of_range_raises(self, momentum) -> None:
        with pytest.raises(ValueError, match="momentum"):
            Nesterov(momentum=momentum)


class TestReduction:
    def test_zero_momentum_matches_plain_sgd(self) -> None:
        lr = 0.05
        nesterov = Nesterov(lr=lr, momentum=0.0)
        sgd = SGD(lr=lr)
        theta_nes, theta_sgd = np.array([3.0, -1.0]), np.array([3.0, -1.0])

        rng = np.random.default_rng(0)
        for _ in range(20):
            grad = rng.standard_normal(2)
            nesterov.step([theta_nes], [grad.copy()])
            sgd.step([theta_sgd], [grad.copy()])

        np.testing.assert_allclose(theta_nes, theta_sgd)


class TestSubstitutionCorrectness:
    def test_matches_the_literal_lookahead_gradient_formula(self) -> None:
        # docs/derivations/optim.md section 4: our reformulation tracks the
        # lookahead point psi_t = theta_t + mu*v_t directly, using only the
        # gradient at the current tracked point. This test drives an
        # independent, literal implementation of textbook NAG -- which
        # evaluates the gradient at the lookahead point explicitly, using
        # the known closed-form gradient of a toy quadratic -- and checks
        # that our psi trajectory equals the literal method's own
        # reconstructed lookahead point at every step.
        A = np.diag([2.0, 5.0])

        def grad_fn(theta: np.ndarray) -> np.ndarray:
            return A @ theta

        lr, mu = 0.05, 0.8
        psi = np.array([3.0, -2.0])
        theta_lit = psi.copy()
        v_lit = np.zeros(2)

        opt = Nesterov(lr=lr, momentum=mu)

        for _ in range(30):
            lookahead = theta_lit + mu * v_lit
            np.testing.assert_allclose(psi, lookahead, atol=1e-10)

            g = grad_fn(psi)
            opt.step([psi], [g])

            v_lit = mu * v_lit - lr * grad_fn(lookahead)
            theta_lit = theta_lit + v_lit


class TestBehavioral:
    def test_converges_on_a_quadratic_bowl(self) -> None:
        A = np.diag([1.0, 4.0])
        theta = np.array([5.0, -3.0])
        opt = Nesterov(lr=0.1, momentum=0.9)

        for _ in range(200):
            grad = A @ theta
            opt.step([theta], [grad])

        assert np.linalg.norm(theta) < 1e-3

    def test_beats_plain_sgd_on_an_ill_conditioned_quadratic(self) -> None:
        A = np.diag([1.0, 100.0])
        lr = 0.005
        theta_sgd = np.array([10.0, 1.0])
        theta_nes = theta_sgd.copy()
        sgd = SGD(lr=lr)
        nesterov = Nesterov(lr=lr, momentum=0.9)

        for _ in range(100):
            sgd.step([theta_sgd], [A @ theta_sgd])
            nesterov.step([theta_nes], [A @ theta_nes])

        loss_sgd = 0.5 * theta_sgd @ A @ theta_sgd
        loss_nes = 0.5 * theta_nes @ A @ theta_nes
        assert loss_nes < loss_sgd
