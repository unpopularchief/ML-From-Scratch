r"""Scaled dot-product attention, composed from Tensor ops.

No backward is defined here: ``matmul``, ``transpose``, ``softmax`` and the
mask add are all existing differentiable ops, so :meth:`Tensor.backward`
differentiates the whole thing. Full derivation: docs/derivations/attention.md.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scratchgrad.autograd.functional import softmax
from scratchgrad.autograd.tensor import Tensor

# Added to blocked scores. Finite (not -inf) so a fully blocked row stays NaN-free.
_MASK_VALUE = -1e9


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Return ``(output, weights)``, with ``output = A V``.

    :math:`A = \operatorname{softmax}(QK^\top/\sqrt{d_k} + M)` over the key axis.

    Shapes: ``q (..., T_q, d_k)``, ``k (..., T_k, d_k)``, ``v (..., T_k, d_v)``;
    ``output (..., T_q, d_v)`` and ``weights (..., T_q, T_k)``. ``mask`` is a
    boolean array broadcastable to ``(..., T_q, T_k)`` where ``True`` marks
    positions that may be attended to (see ``causal_mask``/``padding_mask``).
    """
    d_k = q.shape[-1]
    if k.shape[-1] != d_k:
        raise ValueError(f"q and k need the same last dim, got {d_k} and {k.shape[-1]}")
    if v.shape[-2] != k.shape[-2]:
        raise ValueError(
            f"k and v need the same length, got {k.shape[-2]} and {v.shape[-2]}"
        )
    swap_last_two = (*range(k.data.ndim - 2), k.data.ndim - 1, k.data.ndim - 2)
    scores = (q @ k.transpose(swap_last_two)) * (1.0 / np.sqrt(d_k))
    if mask is not None:
        scores = scores + np.where(mask, 0.0, _MASK_VALUE)  # constant: no grad path
    weights = softmax(scores)
    return weights @ v, weights
