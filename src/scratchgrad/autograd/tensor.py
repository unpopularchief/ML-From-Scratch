"""Tensor: an ndarray that records the graph needed to differentiate through it.

Full derivation: docs/derivations/autograd.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from scratchgrad.typing import FloatArray

# Maps the upstream gradient dL/dy to one gradient dL/d(parent) per parent.
VJP = Callable[[FloatArray], tuple[FloatArray, ...]]


class Tensor:
    """A float64 ndarray plus the bookkeeping for reverse-mode autodiff.

    Parameters
    ----------
    data : array_like
        Stored as a float64 ndarray (the project dtype policy).
    requires_grad : bool, default=False
        If True, :meth:`backward` accumulates ``dL/d(self)`` into
        :attr:`grad`.

    Attributes
    ----------
    data : ndarray
    grad : ndarray or None
        ``None`` until a :meth:`backward` call reaches this tensor.

    Examples
    --------
    >>> x = Tensor([2.0, 3.0], requires_grad=True)
    >>> (x * x).sum().backward()
    >>> x.grad
    array([4., 6.])

    """

    # Make ``ndarray + Tensor`` defer to Tensor.__radd__ instead of numpy
    # broadcasting the Tensor as an object array.
    __array_ufunc__ = None

    def __init__(self, data: Any, requires_grad: bool = False) -> None:
        """Wrap ``data`` as a float64 array."""
        self.data: FloatArray = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad: FloatArray | None = None
        self._parents: tuple[Tensor, ...] = ()
        self._vjp: VJP | None = None

    @classmethod
    def _from_op(
        cls, data: FloatArray, parents: tuple[Tensor, ...], vjp: VJP
    ) -> Tensor:
        """Build the output node of an op, recording the graph only if needed."""
        out = cls(data, requires_grad=any(p.requires_grad for p in parents))
        if out.requires_grad:
            out._parents = parents
            out._vjp = vjp
        return out

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of :attr:`data`."""
        return self.data.shape

    def __repr__(self) -> str:
        """Show data and the ``requires_grad`` flag."""
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def zero_grad(self) -> None:
        """Reset :attr:`grad` to ``None``."""
        self.grad = None

    def detach(self) -> Tensor:
        """Return a graph-free tensor sharing no history with ``self``."""
        return Tensor(self.data.copy())

    def _accumulate(self, g: FloatArray) -> None:
        # Fan-out: a tensor used k times receives k contributions, summed.
        if self.grad is None:
            self.grad = np.array(g, dtype=np.float64)
        else:
            self.grad = self.grad + g

    def _topological_order(self) -> list[Tensor]:
        """Nodes reachable from ``self`` through grad-requiring parents.

        Post-order depth-first search, so every node appears *after* all
        its parents. Iterative: a long chain (e.g. an unrolled RNN) would
        exceed Python's recursion limit if this recursed.
        """
        order: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._parents:
                if parent.requires_grad and id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad: FloatArray | None = None) -> None:
        """Accumulate ``dL/d(leaf)`` into every reachable leaf's :attr:`grad`.

        Parameters
        ----------
        grad : ndarray, optional
            ``dL/d(self)``. Defaults to 1, which is only valid when
            ``self`` is a scalar.

        Notes
        -----
        Leaf gradients accumulate across calls (call :meth:`zero_grad` to
        reset). Intermediate nodes are reset at the start of each call so a
        second ``backward()`` over the same graph does not double-count.

        """
        if not self.requires_grad:
            raise ValueError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise ValueError(
                    "grad must be given explicitly for a non-scalar tensor, "
                    f"got shape {self.shape}"
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(grad, dtype=np.float64)
            if seed.shape != self.shape:
                raise ValueError(
                    f"grad shape {seed.shape} does not match tensor shape {self.shape}"
                )

        order = self._topological_order()
        for node in order:
            if node._vjp is not None:
                node.grad = None
        self._accumulate(seed)

        for node in reversed(order):
            if node._vjp is None:
                continue
            assert node.grad is not None
            for parent, g in zip(node._parents, node._vjp(node.grad), strict=True):
                if parent.requires_grad:
                    parent._accumulate(g)

    # Operators and methods delegate to ops.py (imported lazily: ops builds
    # Tensors, so a module-level import would be circular).

    def __add__(self, other: Any) -> Tensor:
        """Return ``self + other``."""
        from scratchgrad.autograd import ops

        return ops.add(self, other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> Tensor:
        """Return ``self - other``."""
        from scratchgrad.autograd import ops

        return ops.sub(self, other)

    def __rsub__(self, other: Any) -> Tensor:
        """Return ``other - self``."""
        from scratchgrad.autograd import ops

        return ops.sub(other, self)

    def __mul__(self, other: Any) -> Tensor:
        """Return ``self * other``."""
        from scratchgrad.autograd import ops

        return ops.mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> Tensor:
        """Return ``self / other``."""
        from scratchgrad.autograd import ops

        return ops.div(self, other)

    def __rtruediv__(self, other: Any) -> Tensor:
        """Return ``other / self``."""
        from scratchgrad.autograd import ops

        return ops.div(other, self)

    def __neg__(self) -> Tensor:
        """Return ``-self``."""
        from scratchgrad.autograd import ops

        return ops.neg(self)

    def __pow__(self, exponent: float) -> Tensor:
        """Return ``self ** exponent`` (constant exponent)."""
        from scratchgrad.autograd import ops

        return ops.pow(self, exponent)

    def __matmul__(self, other: Any) -> Tensor:
        """Return ``self @ other``."""
        from scratchgrad.autograd import ops

        return ops.matmul(self, other)

    def __rmatmul__(self, other: Any) -> Tensor:
        """Return ``other @ self``."""
        from scratchgrad.autograd import ops

        return ops.matmul(other, self)

    def __getitem__(self, index: Any) -> Tensor:
        """Return ``self[index]`` with scatter-add gradients for repeated indices."""
        from scratchgrad.autograd import ops

        return ops.getitem(self, index)

    def exp(self) -> Tensor:
        """Elementwise ``e**x``."""
        from scratchgrad.autograd import ops

        return ops.exp(self)

    def log(self) -> Tensor:
        """Elementwise natural log."""
        from scratchgrad.autograd import ops

        return ops.log(self)

    def tanh(self) -> Tensor:
        """Elementwise hyperbolic tangent."""
        from scratchgrad.autograd import ops

        return ops.tanh(self)

    def sigmoid(self) -> Tensor:
        """Elementwise logistic sigmoid."""
        from scratchgrad.autograd import ops

        return ops.sigmoid(self)

    def relu(self) -> Tensor:
        """Elementwise ``max(x, 0)``."""
        from scratchgrad.autograd import ops

        return ops.relu(self)

    def sum(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        """Sum over ``axis`` (all axes if None)."""
        from scratchgrad.autograd import ops

        return ops.sum(self, axis, keepdims)

    def mean(
        self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Tensor:
        """Mean over ``axis`` (all axes if None)."""
        from scratchgrad.autograd import ops

        return ops.mean(self, axis, keepdims)

    def reshape(self, *shape: int) -> Tensor:
        """Reshape without changing the data."""
        from scratchgrad.autograd import ops

        return ops.reshape(self, shape)

    def transpose(self, axes: tuple[int, ...] | None = None) -> Tensor:
        """Permute axes (reverse them if ``axes`` is None)."""
        from scratchgrad.autograd import ops

        return ops.transpose(self, axes)

    @property
    def T(self) -> Tensor:
        """Axes reversed."""
        return self.transpose()
