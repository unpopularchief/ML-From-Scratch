r"""Feed-forward network and Pre-LN residual blocks, composed from Tensor ops.

No backward is defined here: every sublayer is a differentiable composition.
Full derivation: docs/derivations/transformer_block.md sections 3-4.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scratchgrad.attention.masking import causal_mask
from scratchgrad.attention.multi_head import MultiHeadAttention
from scratchgrad.autograd.functional import gelu
from scratchgrad.autograd.layers import Dropout, LayerNorm, Linear, Module
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.utils.validation import check_random_state


class _Container(Module):
    """A module that owns child modules: collects parameters, propagates mode."""

    def _children(self) -> list[Module]:
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        """Parameters of every child module, in order."""
        return [p for m in self._children() for p in m.parameters()]

    def train(self) -> Module:
        """Switch this module and every child to training mode."""
        super().train()
        for m in self._children():
            m.train()
        return self

    def eval(self) -> Module:
        """Switch this module and every child to evaluation mode."""
        super().eval()
        for m in self._children():
            m.eval()
        return self


class FeedForward(_Container):
    r""":math:`\operatorname{GELU}(XW_1 + b_1)W_2 + b_2`, applied to each position.

    Examples
    --------
    >>> ffn = FeedForward(d_model=4, d_ff=16, random_state=0)
    >>> ffn(Tensor(np.ones((2, 3, 4)))).shape
    (2, 3, 4)

    """

    def __init__(
        self, d_model: int, d_ff: int, random_state: int | None = None
    ) -> None:
        """Xavier weights, zero biases; ``d_ff`` is usually ``4 * d_model``."""
        seeds = check_random_state(random_state).integers(2**31, size=2)
        self.fc1 = Linear(
            d_model, d_ff, weight_init="xavier", random_state=int(seeds[0])
        )
        self.fc2 = Linear(
            d_ff, d_model, weight_init="xavier", random_state=int(seeds[1])
        )

    def forward(self, x: Tensor) -> Tensor:
        """Expand to ``d_ff``, apply GELU, project back to ``d_model``."""
        return self.fc2(gelu(self.fc1(x)))

    def _children(self) -> list[Module]:
        return [self.fc1, self.fc2]


class EncoderBlock(_Container):
    r"""Pre-LN self-attention block with a feed-forward sublayer.

    :math:`x \mathrel{+}= \mathrm{MHA}(\mathrm{LN}(x))`, then
    :math:`x \mathrel{+}= \mathrm{FFN}(\mathrm{LN}(x))`, with dropout on each branch.

    Examples
    --------
    >>> block = EncoderBlock(d_model=8, num_heads=2, d_ff=32, random_state=0)
    >>> block(Tensor(np.ones((1, 5, 8)))).shape
    (1, 5, 8)

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        """``dropout`` is the drop probability on each residual branch."""
        seeds = check_random_state(random_state).integers(2**31, size=4)
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, random_state=int(seeds[0]))
        self.drop1 = Dropout(dropout, random_state=int(seeds[1]))
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, random_state=int(seeds[2]))
        self.drop2 = Dropout(dropout, random_state=int(seeds[3]))

    def forward(self, x: Tensor, mask: npt.NDArray[np.bool_] | None = None) -> Tensor:
        """Self-attend under ``mask`` (``True`` = may attend), then feed forward."""
        h = self.ln1(x)
        x = x + self.drop1(self.attn(h, mask=mask)[0])
        return x + self.drop2(self.ffn(self.ln2(x)))

    def __call__(  # type: ignore[override]
        self, x: Tensor, mask: npt.NDArray[np.bool_] | None = None
    ) -> Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x, mask)

    def _children(self) -> list[Module]:
        return [self.ln1, self.attn, self.drop1, self.ln2, self.ffn, self.drop2]


class DecoderBlock(_Container):
    r"""Pre-LN block with causal self-attention and optional cross-attention.

    With ``cross_attention=True`` it also attends to an encoder ``memory`` between
    the self-attention and feed-forward sublayers (encoder-decoder). With
    ``cross_attention=False`` it has no cross-attention parameters at all, which
    is the decoder-only (GPT) block.

    Examples
    --------
    >>> block = DecoderBlock(8, 2, 32, cross_attention=False, random_state=0)
    >>> block(Tensor(np.ones((1, 5, 8)))).shape
    (1, 5, 8)

    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        cross_attention: bool = True,
        random_state: int | None = None,
    ) -> None:
        """See :class:`EncoderBlock`; ``cross_attention`` adds the memory sublayer."""
        seeds = check_random_state(random_state).integers(2**31, size=7)
        self.cross_attention = cross_attention
        self.ln1 = LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(
            d_model, num_heads, random_state=int(seeds[0])
        )
        self.drop1 = Dropout(dropout, random_state=int(seeds[1]))
        if cross_attention:
            self.ln_cross = LayerNorm(d_model)
            self.cross_attn = MultiHeadAttention(
                d_model, num_heads, random_state=int(seeds[2])
            )
            self.drop_cross = Dropout(dropout, random_state=int(seeds[3]))
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, random_state=int(seeds[4]))
        self.drop2 = Dropout(dropout, random_state=int(seeds[5]))

    def forward(
        self,
        x: Tensor,
        memory: Tensor | None = None,
        self_mask: npt.NDArray[np.bool_] | None = None,
        memory_mask: npt.NDArray[np.bool_] | None = None,
    ) -> Tensor:
        """``self_mask`` defaults to causal; ``memory`` is needed to cross-attend."""
        if self.cross_attention and memory is None:
            raise ValueError("memory is required when cross_attention=True")
        if not self.cross_attention and memory is not None:
            raise ValueError("memory was given but cross_attention=False")
        if self_mask is None:
            self_mask = causal_mask(x.shape[1], x.shape[1])
        h = self.ln1(x)
        x = x + self.drop1(self.self_attn(h, mask=self_mask)[0])
        if self.cross_attention:
            h = self.ln_cross(x)
            x = x + self.drop_cross(self.cross_attn(h, memory, mask=memory_mask)[0])
        return x + self.drop2(self.ffn(self.ln2(x)))

    def __call__(  # type: ignore[override]
        self,
        x: Tensor,
        memory: Tensor | None = None,
        self_mask: npt.NDArray[np.bool_] | None = None,
        memory_mask: npt.NDArray[np.bool_] | None = None,
    ) -> Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x, memory, self_mask, memory_mask)

    def _children(self) -> list[Module]:
        children: list[Module] = [self.ln1, self.self_attn, self.drop1]
        if self.cross_attention:
            children += [self.ln_cross, self.cross_attn, self.drop_cross]
        return children + [self.ln2, self.ffn, self.drop2]
