"""Tensor optimizers: parity with the array optimizers, AdamW decay, validation."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.autograd import Tensor
from scratchgrad.autograd.optim import SGD, Adam, AdamW
from scratchgrad.optim import SGD as ArraySGD
from scratchgrad.optim import Adam as ArrayAdam

A = np.diag([1.0, 5.0, 20.0])
START = np.array([1.0, -2.0, 3.0])


def run_tensor(opt_cls, steps: int = 30, **kw) -> np.ndarray:
    w = Tensor(START.copy(), requires_grad=True)  # Tensor wraps, not copies
    opt = opt_cls([w], **kw)
    for _ in range(steps):
        opt.zero_grad()
        Aw = (Tensor(A) @ w.reshape(3, 1)).reshape(3)
        ((w * Aw).sum() * 0.5).backward()  # L = w^T A w / 2, so dL/dw = A w
        opt.step()
    return w.data


def run_array(opt, steps: int = 30) -> np.ndarray:
    w = START.copy()
    for _ in range(steps):
        opt.step([w], [A @ w])
    return w


class TestParityWithArrayOptim:
    def test_sgd(self) -> None:
        np.testing.assert_allclose(
            run_tensor(SGD, lr=0.02), run_array(ArraySGD(lr=0.02)), atol=1e-12
        )

    def test_adam(self) -> None:
        np.testing.assert_allclose(
            run_tensor(Adam, lr=0.1), run_array(ArrayAdam(lr=0.1)), atol=1e-10
        )


class TestAdamW:
    def test_zero_decay_equals_adam(self) -> None:
        np.testing.assert_allclose(
            run_tensor(AdamW, lr=0.1, weight_decay=0.0), run_tensor(Adam, lr=0.1)
        )

    def test_decay_is_decoupled_from_the_gradient(self) -> None:
        # With a zero gradient Adam's step is exactly 0, so the whole update
        # is the direct shrink theta * (1 - lr * lambda).
        w = Tensor([2.0, -4.0], requires_grad=True)
        w.grad = np.zeros(2)
        AdamW([w], lr=0.1, weight_decay=0.5).step()
        np.testing.assert_allclose(w.data, np.array([2.0, -4.0]) * (1 - 0.1 * 0.5))


class TestCommon:
    def test_skips_params_without_grad(self) -> None:
        used = Tensor([1.0], requires_grad=True)
        unused = Tensor([5.0], requires_grad=True)
        (used * used).sum().backward()
        for opt in (SGD([used, unused]), Adam([used, unused]), AdamW([used, unused])):
            opt.step()
            np.testing.assert_array_equal(unused.data, [5.0])

    def test_zero_grad_resets_params(self) -> None:
        w = Tensor([1.0], requires_grad=True)
        (w * w).sum().backward()
        opt = SGD([w])
        opt.zero_grad()
        assert w.grad is None

    def test_base_step_is_abstract(self) -> None:
        from scratchgrad.autograd.optim import Optimizer

        with pytest.raises(NotImplementedError):
            Optimizer([], lr=0.1).step()

    @pytest.mark.parametrize(
        ("cls", "kwargs", "match"),
        [
            (SGD, {"lr": 0.0}, "lr"),
            (Adam, {"beta1": 1.0}, "beta1"),
            (Adam, {"beta2": 0.0}, "beta2"),
            (Adam, {"eps": 0.0}, "eps"),
            (AdamW, {"weight_decay": -1.0}, "weight_decay"),
        ],
    )
    def test_validation(self, cls, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            cls([], **kwargs)
