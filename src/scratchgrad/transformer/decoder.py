r"""Decoder stack: ``num_layers`` Pre-LN decoder blocks and a final LayerNorm.

Full derivation: docs/derivations/transformer_block.md section 5.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scratchgrad.autograd.layers import LayerNorm, Module
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.transformer.block import DecoderBlock, _Container
from scratchgrad.utils.validation import check_random_state


class Decoder(_Container):
    """Causal stack over already-embedded ``(B, T, d_model)`` inputs.

    ``cross_attention=True`` is the encoder-decoder decoder (pass the encoder
    output as ``memory``); ``cross_attention=False`` is the decoder-only stack
    a GPT uses.

    Examples
    --------
    >>> dec = Decoder(2, 8, 2, 32, cross_attention=False, random_state=0)
    >>> dec(Tensor(np.ones((1, 5, 8)))).shape
    (1, 5, 8)

    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        cross_attention: bool = True,
        random_state: int | None = None,
    ) -> None:
        """One independently seeded :class:`DecoderBlock` per layer."""
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        seeds = check_random_state(random_state).integers(2**31, size=num_layers)
        self.cross_attention = cross_attention
        self.blocks = [
            DecoderBlock(
                d_model, num_heads, d_ff, dropout, cross_attention, random_state=int(s)
            )
            for s in seeds
        ]
        self.final_ln = LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory: Tensor | None = None,
        self_mask: npt.NDArray[np.bool_] | None = None,
        memory_mask: npt.NDArray[np.bool_] | None = None,
    ) -> Tensor:
        """Run every block (``self_mask`` defaults to causal), then normalize."""
        for block in self.blocks:
            x = block(x, memory, self_mask, memory_mask)
        return self.final_ln(x)

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
        return [*self.blocks, self.final_ln]
