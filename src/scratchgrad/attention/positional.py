r"""Positional encodings: fixed sinusoidal and learned.

Attention is permutation-equivariant, so order must be injected into the
inputs. Both layers add a ``(T, d_model)`` table to ``(B, T, d_model)``
embeddings. Full derivation: docs/derivations/multi_head_positional.md section 2.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.autograd.layers import Module
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.typing import FloatArray
from scratchgrad.utils.validation import check_random_state


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> FloatArray:
    r"""Return the ``(max_len, d_model)`` table of Vaswani et al. (2017).

    :math:`PE_{p,2i}=\sin(p\,\omega_i)`, :math:`PE_{p,2i+1}=\cos(p\,\omega_i)`
    with :math:`\omega_i = 10000^{-2i/d}`.

    Examples
    --------
    >>> sinusoidal_positional_encoding(2, 4)[0]
    array([0., 1., 0., 1.])

    """
    if max_len <= 0:
        raise ValueError(f"max_len must be positive, got {max_len}")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError(f"d_model must be positive and even, got {d_model}")
    positions = np.arange(max_len)[:, None]
    omega = 10000.0 ** (-np.arange(0, d_model, 2) / d_model)
    table = np.zeros((max_len, d_model))
    table[:, 0::2] = np.sin(positions * omega)
    table[:, 1::2] = np.cos(positions * omega)
    return table


class SinusoidalPositionalEncoding(Module):
    """Add the fixed sinusoidal table: ``x + PE[:T]``. No parameters.

    Examples
    --------
    >>> layer = SinusoidalPositionalEncoding(d_model=4, max_len=8)
    >>> layer(Tensor(np.zeros((2, 3, 4)))).shape
    (2, 3, 4)

    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        """Precompute the table for positions ``0 .. max_len - 1``."""
        self.table = sinusoidal_positional_encoding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Add the first ``T`` rows of the table to ``x`` of shape ``(..., T, d)``."""
        _check_input(x, self.table.shape)
        return x + self.table[: x.shape[-2]]


class LearnedPositionalEncoding(Module):
    r"""Add a trainable ``(max_len, d_model)`` table: ``x + P[:T]``.

    Initialized :math:`\mathcal{N}(0, 0.02^2)` (the GPT-2 choice).

    Examples
    --------
    >>> layer = LearnedPositionalEncoding(d_model=4, max_len=8, random_state=0)
    >>> layer(Tensor(np.zeros((2, 3, 4)))).shape
    (2, 3, 4)

    """

    def __init__(
        self, d_model: int, max_len: int = 5000, random_state: int | None = None
    ) -> None:
        """``random_state`` seeds the initial table."""
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_len <= 0:
            raise ValueError(f"max_len must be positive, got {max_len}")
        rng = check_random_state(random_state)
        self.table = Tensor(
            0.02 * rng.standard_normal((max_len, d_model)), requires_grad=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Add the first ``T`` rows of the table to ``x`` of shape ``(..., T, d)``."""
        _check_input(x, self.table.shape)
        return x + self.table[: x.shape[-2]]

    def parameters(self) -> list[Tensor]:
        """``[table]``."""
        return [self.table]


def _check_input(x: Tensor, table_shape: tuple[int, ...]) -> None:
    max_len, d_model = table_shape
    if x.data.ndim < 2 or x.shape[-1] != d_model:
        raise ValueError(f"x must have shape (..., T, {d_model}), got {x.shape}")
    if x.shape[-2] > max_len:
        raise ValueError(f"sequence length {x.shape[-2]} exceeds max_len={max_len}")
