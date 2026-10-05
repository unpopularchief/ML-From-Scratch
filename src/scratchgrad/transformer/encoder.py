r"""Encoder stack: ``num_layers`` Pre-LN blocks and a final LayerNorm.

Full derivation: docs/derivations/transformer_block.md section 5.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from scratchgrad.autograd.layers import LayerNorm, Module
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.transformer.block import EncoderBlock, _Container
from scratchgrad.utils.validation import check_random_state


class Encoder(_Container):
    """Bidirectional stack over already-embedded ``(B, T, d_model)`` inputs.

    Pre-LN leaves the residual stream unnormalized, so the final ``LayerNorm``
    is what hands normalized features to whatever reads the stack's output.

    Examples
    --------
    >>> enc = Encoder(num_layers=2, d_model=8, num_heads=2, d_ff=32, random_state=0)
    >>> enc(Tensor(np.ones((1, 5, 8)))).shape
    (1, 5, 8)

    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        """One independently seeded :class:`EncoderBlock` per layer."""
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        seeds = check_random_state(random_state).integers(2**31, size=num_layers)
        self.blocks = [
            EncoderBlock(d_model, num_heads, d_ff, dropout, random_state=int(s))
            for s in seeds
        ]
        self.final_ln = LayerNorm(d_model)

    def forward(self, x: Tensor, mask: npt.NDArray[np.bool_] | None = None) -> Tensor:
        """Run every block under ``mask`` (e.g. ``padding_mask``), then normalize."""
        for block in self.blocks:
            x = block(x, mask)
        return self.final_ln(x)

    def __call__(  # type: ignore[override]
        self, x: Tensor, mask: npt.NDArray[np.bool_] | None = None
    ) -> Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x, mask)

    def _children(self) -> list[Module]:
        return [*self.blocks, self.final_ln]
