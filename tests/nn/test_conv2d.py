"""Tests for scratchgrad.nn.layers.Conv2d."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import Conv2d
from scratchgrad.nn.layers.conv2d import _conv2d_backward, _conv2d_forward
from tests.helpers.gradcheck import gradient_check


class TestContract:
    def test_invalid_channels_raise(self) -> None:
        with pytest.raises(ValueError, match="in_channels"):
            Conv2d(0, 2, 3)
        with pytest.raises(ValueError, match="out_channels"):
            Conv2d(2, 0, 3)

    @pytest.mark.parametrize(
        ("keyword", "value"),
        [("kernel_size", 0), ("stride", 0), ("padding", -1)],
    )
    def test_invalid_spatial_parameter_raises(self, keyword: str, value: int) -> None:
        with pytest.raises(ValueError, match=keyword):
            if keyword == "kernel_size":
                Conv2d(1, 2, kernel_size=value)
            else:
                Conv2d(1, 2, kernel_size=3, **{keyword: value})

    def test_unknown_weight_init_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_init"):
            Conv2d(1, 2, 3, weight_init="bogus")

    def test_wrong_input_channel_count_raises(self) -> None:
        layer = Conv2d(2, 3, 3, random_state=0)
        with pytest.raises(ValueError, match="channels"):
            layer.forward(np.ones((1, 1, 5, 5)))

    def test_oversized_kernel_raises(self) -> None:
        layer = Conv2d(1, 2, 4, random_state=0)
        with pytest.raises(ValueError, match="padded input"):
            layer.forward(np.ones((1, 1, 3, 3)))

    def test_parameters_and_grads_match_with_bias(self) -> None:
        layer = Conv2d(2, 3, 3, random_state=0)
        y = layer.forward(np.ones((4, 2, 5, 5)))
        layer.backward(np.ones_like(y))
        assert len(layer.parameters()) == len(layer.grads()) == 2
        for parameter, grad in zip(layer.parameters(), layer.grads(), strict=True):
            assert parameter.shape == grad.shape

    def test_bias_false_exposes_only_weights(self) -> None:
        layer = Conv2d(1, 2, 3, bias=False, random_state=0)
        y = layer.forward(np.ones((1, 1, 4, 4)))
        layer.backward(np.ones_like(y))
        assert layer.b is None
        assert len(layer.parameters()) == len(layer.grads()) == 1


class TestAnalytic:
    def test_forward_matches_hand_computed_patch_sums(self) -> None:
        x = np.arange(1.0, 10.0).reshape(1, 1, 3, 3)
        W = np.ones((1, 1, 2, 2))
        b = np.array([1.0])

        y = _conv2d_forward(x, W, b, (1, 1), (0, 0))

        np.testing.assert_allclose(y, [[[[13.0, 17.0], [25.0, 29.0]]]])

    def test_forward_is_cross_correlation_not_flipped_convolution(self) -> None:
        x = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        W = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        y = _conv2d_forward(x, W, None, (1, 1), (0, 0))
        np.testing.assert_allclose(y, [[[[30.0]]]])

    def test_stride_padding_and_rectangular_kernel_shape(self) -> None:
        layer = Conv2d(
            2,
            4,
            kernel_size=(3, 2),
            stride=(2, 1),
            padding=(1, 0),
            random_state=0,
        )
        assert layer.forward(np.ones((3, 2, 6, 5))).shape == (3, 4, 3, 4)

    def test_backward_sums_overlapping_input_contributions(self) -> None:
        x = np.ones((1, 1, 3, 3))
        W = np.ones((1, 1, 2, 2))
        grad_output = np.ones((1, 1, 2, 2))

        dx, dW, db = _conv2d_backward(x, W, grad_output, (1, 1), (0, 0))

        np.testing.assert_allclose(dx, [[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]])
        np.testing.assert_allclose(dW, np.full((1, 1, 2, 2), 4.0))
        np.testing.assert_allclose(db, [4.0])


class TestGradient:
    def test_dX_matches_finite_differences(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((1, 2, 4, 5))
        W = rng.standard_normal((3, 2, 2, 3))
        b = rng.standard_normal(3)
        R = rng.standard_normal((1, 3, 3, 5))
        dx, _, _ = _conv2d_backward(x, W, R, (1, 1), (0, 1))
        gradient_check(
            lambda x_: float(np.sum(_conv2d_forward(x_, W, b, (1, 1), (0, 1)) * R)),
            dx,
            x,
        )

    def test_dW_matches_finite_differences(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((2, 1, 4, 4))
        W = rng.standard_normal((2, 1, 3, 2))
        b = rng.standard_normal(2)
        R = rng.standard_normal((2, 2, 2, 3))
        _, dW, _ = _conv2d_backward(x, W, R, (1, 1), (0, 0))
        gradient_check(
            lambda W_: float(np.sum(_conv2d_forward(x, W_, b, (1, 1), (0, 0)) * R)),
            dW,
            W,
        )

    def test_db_matches_finite_differences(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((2, 1, 4, 4))
        W = rng.standard_normal((2, 1, 2, 2))
        b = rng.standard_normal(2)
        R = rng.standard_normal((2, 2, 2, 2))
        _, _, db = _conv2d_backward(x, W, R, (2, 2), (0, 0))
        gradient_check(
            lambda b_: float(np.sum(_conv2d_forward(x, W, b_, (2, 2), (0, 0)) * R)),
            db,
            b,
        )


class TestDeterminism:
    def test_same_random_state_gives_identical_weights(self) -> None:
        a = Conv2d(2, 3, (2, 3), random_state=0)
        b = Conv2d(2, 3, (2, 3), random_state=0)
        np.testing.assert_array_equal(a.W, b.W)

    def test_zeros_initialization(self) -> None:
        layer = Conv2d(2, 3, 3, weight_init="zeros")
        np.testing.assert_array_equal(layer.W, np.zeros((3, 2, 3, 3)))
