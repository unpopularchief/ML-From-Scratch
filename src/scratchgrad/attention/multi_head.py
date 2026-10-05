r"""Multi-head attention, composed from Tensor ops and ``Linear``.

No backward is defined here: reshape, transpose, matmul and softmax are all
differentiable ops. Full derivation: docs/derivations/multi_head_positional.md
section 1.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scratchgrad.attention.scaled_dot_product import scaled_dot_product_attention
from scratchgrad.autograd.layers import Linear, Module
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.utils.validation import check_random_state


class MultiHeadAttention(Module):
    r"""``h`` attention heads over ``d_model / h``-dim slices, then an output map.

    :math:`\text{out} = \operatorname{concat}_i\,\operatorname{attn}(QW^Q_i,
    KW^K_i, VW^V_i)\,W^O`, implemented with full ``d_model x d_model``
    projections whose columns are split into heads. Weights are Xavier
    uniform, biases zero.

    Examples
    --------
    >>> mha = MultiHeadAttention(d_model=8, num_heads=2, random_state=0)
    >>> out, weights = mha(Tensor(np.ones((1, 5, 8))))
    >>> out.shape, weights.shape
    ((1, 5, 8), (1, 2, 5, 5))

    """

    def __init__(
        self, d_model: int, num_heads: int, random_state: int | None = None
    ) -> None:
        """``d_model`` must be divisible by ``num_heads``."""
        if d_model <= 0 or num_heads <= 0:
            raise ValueError(
                f"d_model and num_heads must be positive, got {d_model}, {num_heads}"
            )
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_heads={num_heads}"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        seeds = check_random_state(random_state).integers(2**31, size=4)
        self.w_q, self.w_k, self.w_v, self.w_o = (
            Linear(d_model, d_model, weight_init="xavier", random_state=int(s))
            for s in seeds
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        """``(B, T, d)`` -> ``(B, h, T, d/h)``."""
        b, t, _ = x.shape
        h = self.num_heads
        return x.reshape(b, t, h, self.d_model // h).transpose((0, 2, 1, 3))

    def forward(
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(output, weights)``: ``(B, T_q, d)`` and ``(B, h, T_q, T_k)``.

        ``k`` defaults to ``q`` and ``v`` to ``k`` (self-attention). ``mask`` is
        boolean, ``True`` = may attend: ``(T_q, T_k)`` or ``(B, 1, T_k)`` (as
        from ``causal_mask``/``padding_mask``, combined with ``&``) or already
        ``(B, h, T_q, T_k)``.
        """
        k = q if k is None else k
        v = k if v is None else v
        for name, x in (("q", q), ("k", k), ("v", v)):
            if x.data.ndim != 3 or x.shape[-1] != self.d_model:
                raise ValueError(
                    f"{name} must have shape (B, T, {self.d_model}), got {x.shape}"
                )
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]  # insert the head axis: (B, ., T_k) -> (B, 1, ., T_k)
        out, weights = scaled_dot_product_attention(
            self._split_heads(self.w_q(q)),
            self._split_heads(self.w_k(k)),
            self._split_heads(self.w_v(v)),
            mask,
        )
        b, t_q = q.shape[:2]
        merged = out.transpose((0, 2, 1, 3)).reshape(b, t_q, self.d_model)
        return self.w_o(merged), weights

    def __call__(  # type: ignore[override]
        self,
        q: Tensor,
        k: Tensor | None = None,
        v: Tensor | None = None,
        mask: npt.NDArray[np.bool_] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Alias for :meth:`forward`."""
        return self.forward(q, k, v, mask)

    def parameters(self) -> list[Tensor]:
        """``W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o``."""
        return [
            p
            for lin in (self.w_q, self.w_k, self.w_v, self.w_o)
            for p in lin.parameters()
        ]
