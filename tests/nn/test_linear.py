"""Tests for scratchgrad.nn.layers.Linear.

Tiers (plan.md section 3, docs/derivations/nn.md section 6): a
hand-computed forward pass, gradient checks for dX/dW/db against
_linear_forward via a fixed random upstream direction, the
parameters()/grads() contract, invalid-hyperparameter errors, and
determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import Linear
from scratchgrad.nn.layers.linear import _linear_backward, _linear_forward
from tests.helpers.gradcheck import gradient_check


class TestContract:
    def test_non_positive_in_features_raises(self) -> None:
        with pytest.raises(ValueError, match="in_features"):
            Linear(0, 3)
        with pytest.raises(ValueError, match="in_features"):
            Linear(-1, 3)

    def test_non_positive_out_features_raises(self) -> None:
        with pytest.raises(ValueError, match="out_features"):
            Linear(3, 0)

    def test_unknown_weight_init_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_init"):
            Linear(3, 2, weight_init="bogus")

    def test_parameters_and_grads_same_order_and_shape(self) -> None:
        layer = Linear(4, 3, random_state=0)
        x = np.ones((5, 4))
        layer.forward(x)
        layer.backward(np.ones((5, 3)))

        params, grads = layer.parameters(), layer.grads()
        assert len(params) == len(grads) == 2
        for p, g in zip(params, grads, strict=True):
            assert p.shape == g.shape

    def test_bias_starts_at_zero(self) -> None:
        layer = Linear(4, 3, random_state=0)
        np.testing.assert_array_equal(layer.b, np.zeros(3))

    def test_zeros_weight_init_starts_W_at_zero(self) -> None:
        layer = Linear(4, 3, weight_init="zeros")
        np.testing.assert_array_equal(layer.W, np.zeros((4, 3)))


class TestAnalytic:
    def test_forward_matches_hand_computation(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, -1.0])
        np.testing.assert_allclose(_linear_forward(X, W, b), [[2.0, 1.0], [4.0, 3.0]])

    def test_backward_matches_hand_computation(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        grad_output = np.array([[1.0, 1.0], [1.0, 1.0]])

        dX, dW, db = _linear_backward(X, W, grad_output)

        np.testing.assert_allclose(dX, grad_output @ W.T)
        np.testing.assert_allclose(dW, X.T @ grad_output)
        np.testing.assert_allclose(db, [2.0, 2.0])


class TestGradient:
    def test_dX_matches_finite_differences(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((5, 4))
        W = rng.standard_normal((4, 3))
        b = rng.standard_normal(3)
        R = rng.standard_normal((5, 3))  # fixed upstream gradient direction

        dX, _, _ = _linear_backward(X, W, R)
        gradient_check(lambda X_: float(np.sum(_linear_forward(X_, W, b) * R)), dX, X)

    def test_dW_matches_finite_differences(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((5, 4))
        W = rng.standard_normal((4, 3))
        b = rng.standard_normal(3)
        R = rng.standard_normal((5, 3))

        _, dW, _ = _linear_backward(X, W, R)
        gradient_check(lambda W_: float(np.sum(_linear_forward(X, W_, b) * R)), dW, W)

    def test_db_matches_finite_differences(self, rng: np.random.Generator) -> None:
        X = rng.standard_normal((5, 4))
        W = rng.standard_normal((4, 3))
        b = rng.standard_normal(3)
        R = rng.standard_normal((5, 3))

        _, _, db = _linear_backward(X, W, R)
        gradient_check(lambda b_: float(np.sum(_linear_forward(X, W, b_) * R)), db, b)


class TestDeterminism:
    def test_same_random_state_gives_byte_identical_weights(self) -> None:
        a = Linear(4, 3, random_state=0)
        b = Linear(4, 3, random_state=0)
        np.testing.assert_array_equal(a.W, b.W)

    def test_forward_backward_do_not_mutate_hyperparameters(self) -> None:
        layer = Linear(4, 3, random_state=0, weight_init="xavier")
        layer.forward(np.ones((5, 4)))
        layer.backward(np.ones((5, 3)))
        assert layer.in_features == 4
        assert layer.out_features == 3
        assert layer.weight_init == "xavier"
