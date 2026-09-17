"""Tests for scratchgrad.nn.layers.Dropout.

Tiers (plan.md section 3, docs/derivations/dropout_batchnorm.md section 4):
a gradient check against a fixed mask, the inverted-dropout E[mask]=1
reduction identity, the eval-mode identity contract, invalid-p errors, and
determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import Dropout
from scratchgrad.nn.layers.dropout import (
    _dropout_backward,
    _dropout_forward,
    _dropout_mask,
)
from tests.helpers.gradcheck import gradient_check


class TestContract:
    def test_p_equal_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="p"):
            Dropout(p=1.0)

    def test_negative_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p"):
            Dropout(p=-0.1)

    def test_parameters_and_grads_are_empty(self) -> None:
        layer = Dropout(p=0.5, random_state=0)
        assert layer.parameters() == []
        assert layer.grads() == []

    def test_defaults_to_training_mode(self) -> None:
        assert Dropout().training is True

    def test_eval_forward_is_identity(self) -> None:
        layer = Dropout(p=0.5, random_state=0).eval()
        x = np.ones((5, 4))
        np.testing.assert_array_equal(layer.forward(x), x)

    def test_eval_backward_is_identity(self) -> None:
        layer = Dropout(p=0.5, random_state=0).eval()
        grad_output = np.arange(20.0).reshape(5, 4)
        layer.forward(np.ones((5, 4)))
        np.testing.assert_array_equal(layer.backward(grad_output), grad_output)

    def test_eval_mode_never_draws_from_the_generator(self) -> None:
        x = np.ones((10, 4))
        untouched = Dropout(p=0.5, random_state=0)
        exercised = Dropout(p=0.5, random_state=0)

        exercised.eval()
        for _ in range(5):
            exercised.forward(x)  # should not consume any randomness
        exercised.train()

        # if eval() drew from the generator, this first training draw would
        # diverge from a generator that skipped straight to training
        np.testing.assert_array_equal(exercised.forward(x), untouched.forward(x))


class TestAnalytic:
    def test_train_forward_zeros_or_scales_every_entry(self) -> None:
        layer = Dropout(p=0.5, random_state=0)
        x = np.ones((100, 20))
        y = layer.forward(x)
        # every surviving entry is scaled by 1/(1-p) = 2.0, every dropped one is 0
        assert np.all((y == 0.0) | (y == 2.0))

    def test_train_backward_matches_forward_mask(self) -> None:
        layer = Dropout(p=0.5, random_state=0)
        x = np.ones((10, 5))
        y = layer.forward(x)
        grad_output = np.ones((10, 5))
        dx = layer.backward(grad_output)
        # dL/dx = grad_output * mask, and y = x * mask with x all-ones, so
        # the surviving entries of dx exactly match the surviving entries of y
        np.testing.assert_array_equal(dx, y)


class TestGradient:
    def test_dx_matches_finite_differences(self, rng: np.random.Generator) -> None:
        x = rng.standard_normal((6, 5))
        R = rng.standard_normal((6, 5))
        mask = _dropout_mask(x.shape, p=0.3, rng=rng)

        dx = _dropout_backward(mask, R)
        gradient_check(lambda x_: float(np.sum(_dropout_forward(x_, mask) * R)), dx, x)


class TestReductionIdentity:
    def test_mask_mean_is_approximately_one(self, rng: np.random.Generator) -> None:
        p = 0.4
        mask = _dropout_mask((200_000,), p, rng)
        assert mask.mean() == pytest.approx(1.0, abs=0.01)

    def test_training_output_mean_approximately_preserves_input(
        self, rng: np.random.Generator
    ) -> None:
        layer = Dropout(p=0.4, random_state=0)
        x = np.full((200_000,), 3.0)
        y = layer.forward(x)
        assert y.mean() == pytest.approx(3.0, abs=0.03)


class TestDeterminism:
    def test_same_random_state_gives_byte_identical_masks(self) -> None:
        a = Dropout(p=0.5, random_state=0)
        b = Dropout(p=0.5, random_state=0)
        x = np.ones((10, 4))
        np.testing.assert_array_equal(a.forward(x), b.forward(x))

    def test_successive_calls_draw_new_masks(self) -> None:
        layer = Dropout(p=0.5, random_state=0)
        x = np.ones((20, 20))
        first = layer.forward(x)
        second = layer.forward(x)
        assert not np.array_equal(first, second)

    def test_forward_does_not_mutate_hyperparameters(self) -> None:
        layer = Dropout(p=0.3, random_state=0)
        layer.forward(np.ones((5, 4)))
        assert layer.p == 0.3
