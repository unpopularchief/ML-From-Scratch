"""Tests for scratchgrad.nn.layers.Flatten."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import Flatten
from tests.helpers.gradcheck import gradient_check


def test_forward_preserves_batch_and_flattens_features() -> None:
    x = np.arange(48.0).reshape(2, 3, 2, 4)
    y = Flatten().forward(x)
    assert y.shape == (2, 24)
    np.testing.assert_array_equal(y[0], np.arange(24.0))


def test_backward_restores_original_shape() -> None:
    x = np.ones((2, 3, 4, 5))
    layer = Flatten()
    y = layer.forward(x)
    grad_output = np.arange(y.size, dtype=np.float64).reshape(y.shape)
    dx = layer.backward(grad_output)
    assert dx.shape == x.shape
    np.testing.assert_array_equal(dx.reshape(y.shape), grad_output)


def test_backward_matches_finite_differences(rng: np.random.Generator) -> None:
    x = rng.standard_normal((2, 2, 2, 3))
    R = rng.standard_normal((2, 12))
    layer = Flatten()
    layer.forward(x)
    dx = layer.backward(R)
    gradient_check(
        lambda x_: float(np.sum(x_.reshape(2, -1) * R)),
        dx,
        x,
    )


def test_one_dimensional_input_raises() -> None:
    with pytest.raises(ValueError, match="batch"):
        Flatten().forward(np.ones(4))


def test_wrong_backward_shape_raises() -> None:
    layer = Flatten()
    layer.forward(np.ones((2, 3, 4)))
    with pytest.raises(ValueError, match="grad_output"):
        layer.backward(np.ones((2, 11)))


def test_layer_has_no_parameters() -> None:
    layer = Flatten()
    assert layer.parameters() == []
    assert layer.grads() == []
