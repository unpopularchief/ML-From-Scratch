"""Tiny decoder-only language model built from existing autograd layers."""

from __future__ import annotations

import numpy as np

from scratchgrad.attention.positional import LearnedPositionalEncoding
from scratchgrad.autograd.layers import Embedding, Linear, Module
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.transformer.block import _Container
from scratchgrad.transformer.decoder import Decoder
from scratchgrad.utils.validation import check_random_state


class TinyGPT(_Container):
    """Predict next-token logits with token and position embeddings.

    Inputs are integer IDs of shape ``(batch, time)``. The output has shape
    ``(batch, time, vocab_size)``. The decoder applies its causal mask.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        random_state: int | None = None,
    ) -> None:
        """Construct an untied output head and a decoder-only stack."""
        if context_length <= 0:
            raise ValueError(f"context_length must be positive, got {context_length}")
        seeds = check_random_state(random_state).integers(2**31, size=4)
        self.context_length = context_length
        self.token_embedding = Embedding(vocab_size, d_model, int(seeds[0]))
        self.position = LearnedPositionalEncoding(
            d_model, context_length, int(seeds[1])
        )
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            d_ff,
            dropout,
            cross_attention=False,
            random_state=int(seeds[2]),
        )
        self.head = Linear(
            d_model, vocab_size, weight_init="xavier", random_state=int(seeds[3])
        )

    def forward(self, token_ids: np.ndarray) -> Tensor:  # type: ignore[override]
        """Return causal next-token logits at each input position."""
        token_ids = np.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must have shape (B, T), got {token_ids.shape}")
        hidden = self.position(self.token_embedding(token_ids))
        return self.head(self.decoder(hidden))

    def __call__(self, token_ids: np.ndarray) -> Tensor:  # type: ignore[override]
        """Alias for :meth:`forward`."""
        return self.forward(token_ids)

    def _children(self) -> list[Module]:
        return [self.token_embedding, self.position, self.decoder, self.head]
