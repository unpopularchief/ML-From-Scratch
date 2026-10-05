"""Gradient checks for every autograd op's VJP (docs/derivations/autograd.md)."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.autograd import Tensor
from scratchgrad.autograd.ops import _unbroadcast
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)


def check_unary(op, x: np.ndarray) -> None:
    """Check d/dx sum(op(x) * R) for a fixed random direction R."""
    R = rng.standard_normal(op(Tensor(x)).shape)

    def f(x_: np.ndarray) -> float:
        return float((op(Tensor(x_)).data * R).sum())

    t = Tensor(x, requires_grad=True)
    (op(t) * R).sum().backward()
    gradient_check(f, t.grad, x)


def check_binary(op, a: np.ndarray, b: np.ndarray) -> None:
    R = rng.standard_normal(op(Tensor(a), Tensor(b)).shape)
    ta, tb = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
    (op(ta, tb) * R).sum().backward()
    gradient_check(
        lambda a_: float((op(Tensor(a_), Tensor(b)).data * R).sum()), ta.grad, a
    )
    gradient_check(
        lambda b_: float((op(Tensor(a), Tensor(b_)).data * R).sum()), tb.grad, b
    )


X = rng.standard_normal((3, 4))
POS = rng.uniform(0.5, 2.0, (3, 4))


@pytest.mark.parametrize(
    "op, x",
    [
        (lambda t: -t, X),
        (lambda t: t**3, X),
        (lambda t: t**0.5, POS),
        (lambda t: t**-2, POS),
        (lambda t: t.exp(), X),
        (lambda t: t.log(), POS),
        (lambda t: t.tanh(), X),
        (lambda t: t.sigmoid(), X),
        (lambda t: t.relu(), X + 0.05),  # keep away from the kink at 0
        (lambda t: t.sum(), X),
        (lambda t: t.sum(axis=0), X),
        (lambda t: t.sum(axis=1, keepdims=True), X),
        (lambda t: t.mean(), X),
        (lambda t: t.mean(axis=(0, 1)), X),
        (lambda t: t.mean(axis=1), X),
        (lambda t: t.reshape(2, 6), X),
        (lambda t: t.reshape(12), X),
        (lambda t: t.T, X),
        (lambda t: t.transpose((2, 0, 1)), rng.standard_normal((2, 3, 4))),
    ],
)
def test_unary_vjp(op, x):
    check_unary(op, x)


@pytest.mark.parametrize(
    "op",
    [
        lambda a, b: a + b,
        lambda a, b: a - b,
        lambda a, b: a * b,
        lambda a, b: a / b,
    ],
)
@pytest.mark.parametrize(
    "shape_a, shape_b",
    [((3, 4), (3, 4)), ((3, 4), (4,)), ((3, 1), (1, 4)), ((3, 4), ()), ((4,), (3, 4))],
)
def test_binary_vjp_with_broadcasting(op, shape_a, shape_b):
    check_binary(op, rng.standard_normal(shape_a), rng.uniform(0.5, 2.0, shape_b))


def test_matmul_vjp():
    check_binary(
        lambda a, b: a @ b, rng.standard_normal((3, 4)), rng.standard_normal((4, 2))
    )


@pytest.mark.parametrize(
    "shape_a, shape_b",
    [
        ((2, 3, 4), (2, 4, 5)),  # matching batch
        ((2, 3, 4), (4, 5)),  # shared right operand: grad sums over the batch
        ((3, 4), (2, 4, 5)),  # shared left operand
        ((2, 1, 3, 4), (1, 5, 4, 2)),  # batch dims broadcast against each other
    ],
)
def test_batched_matmul_vjp(shape_a, shape_b):
    check_binary(
        lambda a, b: a @ b, rng.standard_normal(shape_a), rng.standard_normal(shape_b)
    )


def test_batched_matmul_matches_numpy():
    a, b = rng.standard_normal((2, 3, 4)), rng.standard_normal((2, 4, 5))
    np.testing.assert_allclose((Tensor(a) @ Tensor(b)).data, a @ b)


def test_matmul_rejects_non_2d_operands():
    with pytest.raises(ValueError, match="at least 2 dims"):
        Tensor(np.ones(3)) @ Tensor(np.ones((3, 2)))


@pytest.mark.parametrize(
    "index",
    [np.s_[1:3], np.s_[:, 2], np.s_[1, ::2], np.s_[..., :2], np.s_[0, 1]],
)
def test_getitem_vjp_and_matches_numpy(index):
    x = rng.standard_normal((4, 5))
    np.testing.assert_array_equal(Tensor(x)[index].data, x[index])
    check_unary(lambda t: t[index], x)


def test_getitem_gradient_is_zero_outside_the_slice():
    t = Tensor(np.ones((4, 2)), requires_grad=True)
    t[:2].sum().backward()
    np.testing.assert_array_equal(t.grad, [[1, 1], [1, 1], [0, 0], [0, 0]])


def test_pow_rejects_tensor_exponent():
    with pytest.raises(TypeError):
        Tensor([1.0, 2.0]) ** Tensor(2.0)


class TestUnbroadcast:
    def test_leading_and_size_one_axes(self):
        g = np.ones((2, 3, 4))
        assert _unbroadcast(g, (3, 1)).shape == (3, 1)
        np.testing.assert_array_equal(_unbroadcast(g, (3, 1)), np.full((3, 1), 8.0))
        assert _unbroadcast(g, ()).shape == ()

    def test_same_shape_is_identity(self):
        g = rng.standard_normal((3, 4))
        np.testing.assert_array_equal(_unbroadcast(g, (3, 4)), g)


class TestReflectedOperators:
    def test_scalar_on_left(self):
        x = Tensor([2.0, 4.0], requires_grad=True)
        (1.0 - x + 2.0 * x + 8.0 / x + 3.0).sum().backward()
        np.testing.assert_allclose(x.grad, -1.0 + 2.0 - 8.0 / np.array([2.0, 4.0]) ** 2)

    def test_ndarray_on_left_defers_to_tensor(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        out = np.array([3.0, 5.0]) * x
        assert isinstance(out, Tensor)
        out.sum().backward()
        np.testing.assert_array_equal(x.grad, [3.0, 5.0])

    def test_rmatmul(self):
        w = Tensor(np.ones((2, 3)), requires_grad=True)
        out = np.ones((4, 2)) @ w
        assert out.shape == (4, 3)
