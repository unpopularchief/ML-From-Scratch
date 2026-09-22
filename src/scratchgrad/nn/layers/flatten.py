"""Flatten all non-batch dimensions into one feature dimension."""

from __future__ import annotations

import numpy as np

from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray


class Flatten(Module):
    """Reshape ``(N, d1, ..., dk)`` to ``(N, d1 * ... * dk)``.

    The original shape is cached so :meth:`backward` can restore it.
    Flatten has no parameters and does not copy data when NumPy can return
    a view.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Flatten()
    >>> layer.forward(np.ones((2, 3, 4, 5))).shape
    (2, 60)

    """

    def __init__(self) -> None:
        """Initialize an empty shape cache."""
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Preserve the batch axis and flatten all remaining axes."""
        if x.ndim < 2:
            raise ValueError(
                f"x must include a batch and at least one feature axis, got {x.shape}"
            )
        self._input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Restore the input shape cached by :meth:`forward`."""
        flattened = int(np.prod(self._input_shape[1:]))
        expected_shape = (self._input_shape[0], flattened)
        if grad_output.shape != expected_shape:
            raise ValueError(
                f"grad_output must have shape {expected_shape}, got {grad_output.shape}"
            )
        return grad_output.reshape(self._input_shape)
