"""Primitive ops and their vector-Jacobian products (VJPs).

Each op computes ``y = f(inputs)`` and hands :meth:`Tensor._from_op` a
closure mapping the upstream gradient ``g = dL/dy`` to one gradient per
input. Full derivation: docs/derivations/autograd.md section 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scratchgrad.autograd.tensor import Tensor
from scratchgrad.typing import FloatArray
from scratchgrad.utils.math import sigmoid as _sigmoid

Axis = int | tuple[int, ...] | None


def _as_tensor(x: Any) -> Tensor:
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(g: FloatArray, shape: tuple[int, ...]) -> FloatArray:
    """Sum ``g`` down to ``shape``: the VJP of numpy broadcasting."""
    extra = g.ndim - len(shape)
    if extra > 0:
        g = g.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and g.shape[i] != 1)
    if stretched:
        g = g.sum(axis=stretched, keepdims=True)
    return g


def add(a: Any, b: Any) -> Tensor:
    """Add with numpy broadcasting."""
    a, b = _as_tensor(a), _as_tensor(b)
    return Tensor._from_op(
        a.data + b.data,
        (a, b),
        lambda g: (_unbroadcast(g, a.shape), _unbroadcast(g, b.shape)),
    )


def sub(a: Any, b: Any) -> Tensor:
    """Subtract with numpy broadcasting."""
    a, b = _as_tensor(a), _as_tensor(b)
    return Tensor._from_op(
        a.data - b.data,
        (a, b),
        lambda g: (_unbroadcast(g, a.shape), _unbroadcast(-g, b.shape)),
    )


def mul(a: Any, b: Any) -> Tensor:
    """Elementwise ``a * b`` with numpy broadcasting."""
    a, b = _as_tensor(a), _as_tensor(b)
    return Tensor._from_op(
        a.data * b.data,
        (a, b),
        lambda g: (
            _unbroadcast(g * b.data, a.shape),
            _unbroadcast(g * a.data, b.shape),
        ),
    )


def div(a: Any, b: Any) -> Tensor:
    """Elementwise ``a / b`` with numpy broadcasting."""
    a, b = _as_tensor(a), _as_tensor(b)
    return Tensor._from_op(
        a.data / b.data,
        (a, b),
        lambda g: (
            _unbroadcast(g / b.data, a.shape),
            _unbroadcast(-g * a.data / b.data**2, b.shape),
        ),
    )


def neg(a: Tensor) -> Tensor:
    """Negate."""
    return Tensor._from_op(-a.data, (a,), lambda g: (-g,))


def pow(a: Tensor, exponent: float) -> Tensor:  # noqa: A001 - mirrors the ``**`` operator
    """Raise to a constant (non-Tensor) exponent."""
    if isinstance(exponent, Tensor):
        raise TypeError("only constant exponents are supported")
    return Tensor._from_op(
        a.data**exponent,
        (a,),
        lambda g: (g * exponent * a.data ** (exponent - 1),),
    )


def matmul(a: Any, b: Any) -> Tensor:
    """Matrix product ``a @ b`` over the last two axes, batching leading ones.

    Leading (batch) axes broadcast like ``np.matmul``; both operands need at
    least 2 dims (no 1-D vector promotion).
    """
    a, b = _as_tensor(a), _as_tensor(b)
    if a.data.ndim < 2 or b.data.ndim < 2:
        raise ValueError(
            "matmul needs operands with at least 2 dims, "
            f"got {a.data.ndim}-D and {b.data.ndim}-D"
        )
    return Tensor._from_op(
        a.data @ b.data,
        (a, b),
        lambda g: (
            _unbroadcast(g @ np.swapaxes(b.data, -1, -2), a.shape),
            _unbroadcast(np.swapaxes(a.data, -1, -2) @ g, b.shape),
        ),
    )


def exp(a: Tensor) -> Tensor:
    """Elementwise ``e**a``."""
    y = np.exp(a.data)
    return Tensor._from_op(y, (a,), lambda g: (g * y,))


def log(a: Tensor) -> Tensor:
    """Elementwise natural log."""
    return Tensor._from_op(np.log(a.data), (a,), lambda g: (g / a.data,))


def tanh(a: Tensor) -> Tensor:
    """Elementwise hyperbolic tangent."""
    y = np.tanh(a.data)
    return Tensor._from_op(y, (a,), lambda g: (g * (1.0 - y**2),))


def sigmoid(a: Tensor) -> Tensor:
    """Elementwise logistic sigmoid."""
    y = _sigmoid(a.data)
    return Tensor._from_op(y, (a,), lambda g: (g * y * (1.0 - y),))


def relu(a: Tensor) -> Tensor:
    """Elementwise ``max(a, 0)``; the subgradient at 0 is 0."""
    return Tensor._from_op(
        np.maximum(a.data, 0.0), (a,), lambda g: (g * (a.data > 0.0),)
    )


def _expand_reduced(
    g: FloatArray, shape: tuple[int, ...], axis: Axis, keepdims: bool
) -> FloatArray:
    """Broadcast a reduction's gradient back to the reduced input's shape."""
    if axis is not None and not keepdims:
        g = np.expand_dims(g, axis)
    return np.broadcast_to(g, shape)


def sum(a: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:  # noqa: A001
    """Sum over ``axis`` (all axes if None)."""
    return Tensor._from_op(
        a.data.sum(axis=axis, keepdims=keepdims),
        (a,),
        lambda g: (_expand_reduced(g, a.shape, axis, keepdims),),
    )


def mean(a: Tensor, axis: Axis = None, keepdims: bool = False) -> Tensor:
    """Mean over ``axis`` (all axes if None)."""
    y = a.data.mean(axis=axis, keepdims=keepdims)
    count = a.data.size / y.size
    return Tensor._from_op(
        y,
        (a,),
        lambda g: (_expand_reduced(g, a.shape, axis, keepdims) / count,),
    )


def reshape(a: Tensor, shape: tuple[int, ...]) -> Tensor:
    """Reshape; the VJP reshapes the gradient back."""
    return Tensor._from_op(a.data.reshape(shape), (a,), lambda g: (g.reshape(a.shape),))


def getitem(a: Tensor, index: Any) -> Tensor:
    """Index with ints and slices; the VJP scatters the gradient into zeros.

    Advanced (array) indexing is not supported: it can repeat an element, and
    the scatter below assigns rather than accumulates.
    """

    def vjp(g: FloatArray) -> tuple[FloatArray]:
        grad = np.zeros_like(a.data)
        grad[index] = g
        return (grad,)

    return Tensor._from_op(np.array(a.data[index]), (a,), vjp)


def transpose(a: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
    """Permute axes; the VJP applies the inverse permutation."""
    perm = tuple(reversed(range(a.data.ndim))) if axes is None else axes
    inverse = tuple(np.argsort(perm))
    return Tensor._from_op(
        a.data.transpose(perm), (a,), lambda g: (g.transpose(inverse),)
    )
