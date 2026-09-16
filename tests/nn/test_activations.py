"""Tests for scratchgrad.nn.activations (ReLU, Sigmoid, Tanh, Softmax).

Tiers (plan.md section 3, docs/derivations/nn.md section 6): hand-computed
forward values, gradient checks for each _*_backward against its
_*_forward via a fixed random upstream direction, and the Module class
wrappers caching correctly across forward/backward.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.nn import ReLU, Sigmoid, Softmax, Tanh
from scratchgrad.nn.activations import (
    _relu_backward,
    _relu_forward,
    _sigmoid_backward,
    _sigmoid_forward,
    _softmax_backward,
    _softmax_forward,
    _tanh_backward,
    _tanh_forward,
)
from scratchgrad.utils.math import sigmoid, softmax
from tests.helpers.gradcheck import gradient_check


class TestReLU:
    def test_forward_matches_hand_computation(self) -> None:
        x = np.array([-2.0, -0.5, 0.0, 1.5])
        np.testing.assert_allclose(_relu_forward(x), [0.0, 0.0, 0.0, 1.5])

    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        # Away from the x=0 kink, so the analytic subgradient is exact.
        x = rng.standard_normal(10)
        x[np.abs(x) < 0.1] += 1.0  # nudge any near-zero draws off the kink
        R = rng.standard_normal(10)

        analytic = _relu_backward(x, R)
        gradient_check(lambda x_: float(np.sum(_relu_forward(x_) * R)), analytic, x)

    def test_module_forward_backward_round_trip(self) -> None:
        relu = ReLU()
        y = relu.forward(np.array([-1.0, 2.0]))
        grad_x = relu.backward(np.array([1.0, 1.0]))
        np.testing.assert_allclose(y, [0.0, 2.0])
        np.testing.assert_allclose(grad_x, [0.0, 1.0])
        assert relu.parameters() == []
        assert relu.grads() == []


class TestSigmoid:
    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        x = rng.standard_normal(10)
        R = rng.standard_normal(10)
        y = _sigmoid_forward(x)

        analytic = _sigmoid_backward(y, R)
        gradient_check(lambda x_: float(np.sum(_sigmoid_forward(x_) * R)), analytic, x)

    def test_module_matches_utils_math_sigmoid(self) -> None:
        x = np.array([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(Sigmoid().forward(x), sigmoid(x))

    def test_module_forward_backward_round_trip(self) -> None:
        act = Sigmoid()
        x = np.array([-1.0, 0.0, 2.0])
        y = act.forward(x)
        grad_x = act.backward(np.ones(3))
        np.testing.assert_allclose(grad_x, _sigmoid_backward(y, np.ones(3)))


class TestTanh:
    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        x = rng.standard_normal(10)
        R = rng.standard_normal(10)
        y = _tanh_forward(x)

        analytic = _tanh_backward(y, R)
        gradient_check(lambda x_: float(np.sum(_tanh_forward(x_) * R)), analytic, x)

    def test_module_forward_matches_np_tanh(self) -> None:
        x = np.array([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(Tanh().forward(x), np.tanh(x))

    def test_module_forward_backward_round_trip(self) -> None:
        act = Tanh()
        x = np.array([-1.0, 0.0, 2.0])
        y = act.forward(x)
        grad_x = act.backward(np.ones(3))
        np.testing.assert_allclose(grad_x, _tanh_backward(y, np.ones(3)))


class TestSoftmax:
    def test_forward_rows_sum_to_one(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((4, 5))
        y = _softmax_forward(x)
        np.testing.assert_allclose(y.sum(axis=1), np.ones(4))

    def test_backward_matches_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        x = rng.standard_normal((4, 5))
        R = rng.standard_normal((4, 5))
        y = _softmax_forward(x)

        analytic = _softmax_backward(y, R)
        gradient_check(lambda x_: float(np.sum(_softmax_forward(x_) * R)), analytic, x)

    def test_module_matches_utils_math_softmax(self) -> None:
        x = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(Softmax().forward(x), softmax(x, axis=-1))

    def test_module_forward_backward_round_trip(self) -> None:
        act = Softmax()
        x = np.array([[1.0, 2.0, 3.0]])
        y = act.forward(x)
        grad_x = act.backward(np.ones((1, 3)))
        np.testing.assert_allclose(grad_x, _softmax_backward(y, np.ones((1, 3))))
