r"""Elementwise activations: ReLU, Sigmoid, Tanh, Softmax.

Each subclasses ``Module`` with no learnable parameters (``parameters()``/
``grads()`` inherit the empty-list default). Full derivation:
docs/derivations/nn.md section 3.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray
from scratchgrad.utils.math import sigmoid as _sigmoid_fn
from scratchgrad.utils.math import softmax as _softmax_fn


def _relu_forward(x: FloatArray) -> FloatArray:
    """Compute ``max(0, x)``."""
    return np.maximum(0.0, x)


def _relu_backward(x: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Compute ``grad_output * 1[x > 0]`` (subgradient 0 at the kink)."""
    return grad_output * (x > 0)


def _sigmoid_forward(x: FloatArray) -> FloatArray:
    """Compute ``sigma(x)``."""
    return _sigmoid_fn(x)


def _sigmoid_backward(y: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Compute ``grad_output * y * (1 - y)`` from the cached output ``y``."""
    return grad_output * y * (1.0 - y)


def _tanh_forward(x: FloatArray) -> FloatArray:
    """Compute ``tanh(x)``."""
    return np.tanh(x)


def _tanh_backward(y: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Compute ``grad_output * (1 - y**2)`` from the cached output ``y``."""
    return grad_output * (1.0 - y**2)


def _softmax_forward(x: FloatArray) -> FloatArray:
    """Compute ``softmax(x)`` over the last axis."""
    return _softmax_fn(x, axis=-1)


def _softmax_backward(y: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Compute ``J @ grad_output`` per row, ``J = diag(y) - y y^T``.

    ``J @ g = y*g - y*(y . g)`` -- see docs/derivations/nn.md section 3.
    """
    dot = np.sum(y * grad_output, axis=-1, keepdims=True)  # y . grad_output
    return y * (grad_output - dot)


class ReLU(Module):
    r"""Rectified linear unit, :math:`f(x) = \max(0, x)`.

    Examples
    --------
    >>> import numpy as np
    >>> relu = ReLU()
    >>> relu.forward(np.array([-1.0, 0.0, 2.0]))
    array([0., 0., 2.])

    """

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._x: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute ``max(0, x)``, caching ``x`` for :meth:`backward`."""
        self._x = x
        return _relu_forward(x)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dx`` from the upstream gradient ``dL/dy``."""
        return _relu_backward(self._x, grad_output)


class Sigmoid(Module):
    r"""Logistic sigmoid, :math:`f(x) = \sigma(x)`."""

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._y: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute ``sigma(x)``, caching the output for :meth:`backward`."""
        self._y = _sigmoid_forward(x)
        return self._y

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dx`` from the upstream gradient ``dL/dy``."""
        return _sigmoid_backward(self._y, grad_output)


class Tanh(Module):
    r"""Hyperbolic tangent, :math:`f(x) = \tanh(x)`."""

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._y: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute ``tanh(x)``, caching the output for :meth:`backward`."""
        self._y = _tanh_forward(x)
        return self._y

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dx`` from the upstream gradient ``dL/dy``."""
        return _tanh_backward(self._y, grad_output)


class Softmax(Module):
    r"""Softmax over the last axis, :math:`f(x)_i = e^{x_i}/\sum_j e^{x_j}`.

    Rarely used standalone (usually fused inside
    :class:`scratchgrad.nn.losses.CrossEntropyLoss` for numerical
    stability), but implements the full per-sample Jacobian-vector
    product so it stays independently gradient-checkable like every other
    activation here (docs/derivations/nn.md section 3).
    """

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._y: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute ``softmax(x)``, caching the output for :meth:`backward`."""
        self._y = _softmax_forward(x)
        return self._y

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dx`` from the upstream gradient ``dL/dy``."""
        return _softmax_backward(self._y, grad_output)
