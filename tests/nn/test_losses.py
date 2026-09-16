"""Tests for scratchgrad.nn.losses (MSE, BCEWithLogits, CrossEntropy).

Tiers (plan.md section 3, docs/derivations/nn.md section 6): hand-computed
forward values, gradient checks for each _*_backward against its
_*_forward, and the wrapper classes caching correctly across
forward/backward.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from scratchgrad.nn.losses import (
    _bce_with_logits_backward,
    _bce_with_logits_forward,
    _cross_entropy_backward,
    _cross_entropy_forward,
    _mse_backward,
    _mse_forward,
)
from tests.helpers.gradcheck import gradient_check


class TestMSE:
    def test_forward_matches_hand_computation(self) -> None:
        y_pred = np.array([1.0, 2.0])
        y_true = np.array([0.0, 2.0])
        # errors [1, 0] -> mean(1^2, 0^2) = 0.5
        assert _mse_forward(y_pred, y_true) == 0.5

    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        y_pred = rng.standard_normal((5, 2))
        y_true = rng.standard_normal((5, 2))
        analytic = _mse_backward(y_pred, y_true)
        gradient_check(lambda yp: _mse_forward(yp, y_true), analytic, y_pred)

    def test_module_round_trip(self) -> None:
        loss = MSELoss()
        y_pred, y_true = np.array([1.0, 2.0]), np.array([0.0, 2.0])
        value = loss.forward(y_pred, y_true)
        grad = loss.backward()
        assert value == 0.5
        np.testing.assert_allclose(grad, _mse_backward(y_pred, y_true))


class TestBCEWithLogits:
    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        z = rng.standard_normal(8)
        y = rng.integers(0, 2, size=8).astype(np.float64)
        analytic = _bce_with_logits_backward(z, y)
        gradient_check(lambda z_: _bce_with_logits_forward(z_, y), analytic, z)

    def test_matches_naive_sigmoid_cross_entropy_away_from_extremes(self) -> None:
        # For moderate z (no overflow risk), softplus(z) - y*z should equal
        # the naive -[y log sigmoid(z) + (1-y) log(1-sigmoid(z))] form.
        z = np.array([-1.0, 0.5, 2.0])
        y = np.array([0.0, 1.0, 1.0])
        p = 1.0 / (1.0 + np.exp(-z))
        naive = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        assert _bce_with_logits_forward(z, y) == pytest.approx(naive)

    def test_module_round_trip(self) -> None:
        loss = BCEWithLogitsLoss()
        z = np.array([-1.0, 0.5, 2.0])
        y = np.array([0.0, 1.0, 1.0])
        value = loss.forward(z, y)
        grad = loss.backward()
        assert value == _bce_with_logits_forward(z, y)
        np.testing.assert_allclose(grad, _bce_with_logits_backward(z, y))


class TestCrossEntropy:
    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        Z = rng.standard_normal((6, 4))
        labels = rng.integers(0, 4, size=6)
        y_true = np.eye(4)[labels]
        analytic = _cross_entropy_backward(Z, y_true)
        gradient_check(lambda Z_: _cross_entropy_forward(Z_, y_true), analytic, Z)

    def test_module_round_trip(self) -> None:
        loss = CrossEntropyLoss()
        Z = np.array([[10.0, 0.0, 0.0]])  # near-certain class 0
        y_true = np.array([[1.0, 0.0, 0.0]])
        value = loss.forward(Z, y_true)
        grad = loss.backward()
        assert value < 1e-3  # confident, correct prediction -> near-zero loss
        np.testing.assert_allclose(grad, _cross_entropy_backward(Z, y_true))
