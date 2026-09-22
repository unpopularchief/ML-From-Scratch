"""Tests for scratchgrad.nn.layers.MaxPool2d."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import MaxPool2d
from scratchgrad.nn.layers.pooling import (
    _maxpool2d_backward,
    _maxpool2d_forward,
)
from tests.helpers.gradcheck import gradient_check


class TestContract:
    @pytest.mark.parametrize(
        ("keyword", "value"),
        [("kernel_size", 0), ("stride", 0), ("padding", -1)],
    )
    def test_invalid_spatial_parameter_raises(self, keyword: str, value: int) -> None:
        with pytest.raises(ValueError, match=keyword):
            if keyword == "kernel_size":
                MaxPool2d(kernel_size=value)
            else:
                MaxPool2d(kernel_size=2, **{keyword: value})

    def test_excessive_padding_raises(self) -> None:
        with pytest.raises(ValueError, match="padding"):
            MaxPool2d(kernel_size=2, padding=2)

    def test_default_stride_equals_kernel_size(self) -> None:
        layer = MaxPool2d((2, 3))
        assert layer.stride == (2, 3)

    def test_layer_has_no_parameters(self) -> None:
        layer = MaxPool2d(2)
        y = layer.forward(np.ones((1, 1, 4, 4)))
        layer.backward(np.ones_like(y))
        assert layer.parameters() == []
        assert layer.grads() == []


class TestAnalytic:
    def test_forward_matches_hand_computation(self) -> None:
        x = np.arange(16.0).reshape(1, 1, 4, 4)
        y, _ = _maxpool2d_forward(x, (2, 2), (2, 2), (0, 0))
        np.testing.assert_allclose(y, [[[[5.0, 7.0], [13.0, 15.0]]]])

    def test_backward_routes_to_window_winners(self) -> None:
        x = np.arange(16.0).reshape(1, 1, 4, 4)
        _, argmax = _maxpool2d_forward(x, (2, 2), (2, 2), (0, 0))
        dx = _maxpool2d_backward(
            np.ones((1, 1, 2, 2)),
            argmax,
            x.shape,
            (2, 2),
            (2, 2),
            (0, 0),
        )
        expected = np.zeros_like(x)
        expected[0, 0, 1, 1] = 1
        expected[0, 0, 1, 3] = 1
        expected[0, 0, 3, 1] = 1
        expected[0, 0, 3, 3] = 1
        np.testing.assert_array_equal(dx, expected)

    def test_overlapping_windows_accumulate_gradients(self) -> None:
        x = np.zeros((1, 1, 3, 3))
        x[0, 0, 1, 1] = 10.0
        layer = MaxPool2d(kernel_size=2, stride=1)
        y = layer.forward(x)
        dx = layer.backward(np.ones_like(y))
        expected = np.zeros_like(x)
        expected[0, 0, 1, 1] = 4.0
        np.testing.assert_array_equal(dx, expected)

    def test_tie_routes_to_first_maximum(self) -> None:
        x = np.array([[[[2.0, 2.0], [1.0, 0.0]]]])
        layer = MaxPool2d(2)
        y = layer.forward(x)
        dx = layer.backward(np.ones_like(y))
        np.testing.assert_array_equal(dx, [[[[1.0, 0.0], [0.0, 0.0]]]])

    def test_padding_uses_negative_infinity(self) -> None:
        x = -np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
        layer = MaxPool2d(3, stride=1, padding=1)
        y = layer.forward(x)
        assert np.all(y < 0.0)


class TestGradient:
    def test_dX_matches_finite_differences(self) -> None:
        x = np.array([[[[0.1, 1.2, -0.4], [2.3, -1.5, 0.7], [0.2, 1.8, -0.9]]]])
        R = np.array([[[[0.5, -1.0], [1.5, 0.25]]]])
        y, argmax = _maxpool2d_forward(x, (2, 2), (1, 1), (0, 0))
        assert y.shape == R.shape
        dx = _maxpool2d_backward(R, argmax, x.shape, (2, 2), (1, 1), (0, 0))
        gradient_check(
            lambda x_: float(
                np.sum(_maxpool2d_forward(x_, (2, 2), (1, 1), (0, 0))[0] * R)
            ),
            dx,
            x,
        )
