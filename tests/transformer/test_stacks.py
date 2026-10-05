"""Encoder and decoder stacks (docs/derivations/transformer_block.md section 5)."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.attention import padding_mask
from scratchgrad.autograd import Tensor
from scratchgrad.transformer import Decoder, Encoder
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)
B, T, TK, D, H, DFF, L = 2, 4, 3, 8, 2, 16, 2


class TestEncoder:
    def test_stack_then_final_layer_norm(self) -> None:
        enc = Encoder(L, D, H, DFF, random_state=0)
        x = rng.standard_normal((B, T, D))
        h = Tensor(x)
        for blk in enc.blocks:
            h = blk(h)
        np.testing.assert_allclose(
            enc(Tensor(x)).data, enc.final_ln(h).data, atol=1e-12
        )

    def test_output_is_normalized(self) -> None:
        enc = Encoder(L, D, H, DFF, random_state=0)
        out = enc(Tensor(rng.standard_normal((B, T, D))))
        np.testing.assert_allclose(out.data.mean(axis=-1), 0.0, atol=1e-10)

    def test_layers_have_distinct_weights(self) -> None:
        enc = Encoder(L, D, H, DFF, random_state=0)
        w0 = enc.blocks[0].attn.w_q.W.data
        assert not np.allclose(w0, enc.blocks[1].attn.w_q.W.data)

    def test_padding_mask_is_respected(self) -> None:
        enc = Encoder(L, D, H, DFF, random_state=0)
        x = rng.standard_normal((B, T, D))
        x2 = x.copy()
        x2[0, 2:] = rng.standard_normal((T - 2, D))
        mask = padding_mask([2, T], T)
        a = enc(Tensor(x), mask).data
        b = enc(Tensor(x2), mask).data
        np.testing.assert_allclose(a[0, :2], b[0, :2], atol=1e-12)

    def test_parameters_and_gradients(self) -> None:
        enc = Encoder(L, D, H, DFF, random_state=0)
        enc(Tensor(rng.standard_normal((B, T, D)))).sum().backward()
        assert len(enc.parameters()) == L * 16 + 2
        assert all(p.grad is not None for p in enc.parameters())

    def test_gradient_check_input(self) -> None:
        enc = Encoder(L, D, H, DFF, random_state=0)
        x = rng.standard_normal((B, T, D))
        R = rng.standard_normal((B, T, D))
        t = Tensor(x, requires_grad=True)
        (enc(t) * R).sum().backward()
        gradient_check(lambda a: float((enc(Tensor(a)) * R).sum().data), t.grad, x)

    def test_eval_propagates_to_blocks(self) -> None:
        enc = Encoder(L, D, H, DFF, dropout=0.3, random_state=0).eval()
        assert not enc.blocks[1].drop2.training and not enc.final_ln.training

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            Encoder(0, D, H, DFF)


class TestDecoder:
    def test_stack_then_final_layer_norm(self) -> None:
        dec = Decoder(L, D, H, DFF, random_state=0)
        x = rng.standard_normal((B, T, D))
        mem = Tensor(rng.standard_normal((B, TK, D)))
        h = Tensor(x)
        for blk in dec.blocks:
            h = blk(h, mem)
        out = dec(Tensor(x), mem)
        np.testing.assert_allclose(out.data, dec.final_ln(h).data, atol=1e-12)

    def test_decoder_only_is_causal_end_to_end(self) -> None:
        dec = Decoder(L, D, H, DFF, cross_attention=False, random_state=0)
        x = rng.standard_normal((B, T, D))
        x2 = x.copy()
        x2[:, -1] = rng.standard_normal((B, D))
        a, b = dec(Tensor(x)).data, dec(Tensor(x2)).data
        np.testing.assert_allclose(a[:, :-1], b[:, :-1], atol=1e-12)

    def test_parameters_and_gradients(self) -> None:
        dec = Decoder(L, D, H, DFF, random_state=0)
        mem = Tensor(rng.standard_normal((B, TK, D)), requires_grad=True)
        dec(Tensor(rng.standard_normal((B, T, D))), mem).sum().backward()
        assert len(dec.parameters()) == L * 26 + 2
        assert all(p.grad is not None for p in dec.parameters())
        assert mem.grad is not None

    def test_gradient_check_input_decoder_only(self) -> None:
        dec = Decoder(L, D, H, DFF, cross_attention=False, random_state=0)
        x = rng.standard_normal((B, T, D))
        R = rng.standard_normal((B, T, D))
        t = Tensor(x, requires_grad=True)
        (dec(t) * R).sum().backward()
        gradient_check(lambda a: float((dec(Tensor(a)) * R).sum().data), t.grad, x)

    def test_train_eval_propagates(self) -> None:
        dec = Decoder(L, D, H, DFF, dropout=0.3, random_state=0).eval()
        assert not dec.blocks[0].drop_cross.training
        dec.train()
        assert dec.blocks[0].drop_cross.training

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="num_layers"):
            Decoder(0, D, H, DFF)
