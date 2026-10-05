"""Pre-LN blocks (docs/derivations/transformer_block.md sections 3-4)."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.attention import causal_mask, padding_mask
from scratchgrad.autograd import Tensor
from scratchgrad.autograd import functional as F
from scratchgrad.transformer import DecoderBlock, EncoderBlock, FeedForward
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)
B, T, TK, D, H, DFF = 2, 4, 3, 8, 2, 16


def _x(t: int = T) -> np.ndarray:
    return rng.standard_normal((B, t, D))


class TestFeedForward:
    def test_matches_reference(self) -> None:
        ffn = FeedForward(D, DFF, random_state=0)
        x = _x()
        ref = (
            F.gelu(Tensor(x @ ffn.fc1.W.data + ffn.fc1.b.data)).data @ ffn.fc2.W.data
            + ffn.fc2.b.data
        )
        np.testing.assert_allclose(ffn(Tensor(x)).data, ref, atol=1e-12)

    def test_positionwise(self) -> None:
        ffn = FeedForward(D, DFF, random_state=0)
        x = _x()
        full = ffn(Tensor(x)).data
        one = ffn(Tensor(x[:, 1:2])).data
        np.testing.assert_allclose(full[:, 1:2], one, atol=1e-12)

    def test_parameter_count(self) -> None:
        params = FeedForward(D, DFF).parameters()
        assert sum(p.data.size for p in params) == 2 * D * DFF + DFF + D


class TestEncoderBlock:
    def test_pre_ln_structure(self) -> None:
        blk = EncoderBlock(D, H, DFF, random_state=0)
        x = _x()
        mid = x + blk.attn(blk.ln1(Tensor(x)))[0].data
        ref = mid + blk.ffn(blk.ln2(Tensor(mid))).data
        np.testing.assert_allclose(blk(Tensor(x)).data, ref, atol=1e-12)

    def test_zeroed_branches_give_identity(self) -> None:
        blk = EncoderBlock(D, H, DFF, random_state=0)
        blk.attn.w_o.W.data[:] = 0.0
        blk.ffn.fc2.W.data[:] = 0.0
        x = _x()
        np.testing.assert_allclose(blk(Tensor(x)).data, x, atol=1e-12)

    def test_padding_mask_blocks_padded_keys(self) -> None:
        blk = EncoderBlock(D, H, DFF, random_state=0)
        x = _x()
        mask = padding_mask([2, T], T)
        x2 = x.copy()
        x2[0, 2:] = rng.standard_normal((T - 2, D))  # change only padded positions
        a = blk(Tensor(x), mask).data
        b = blk(Tensor(x2), mask).data
        np.testing.assert_allclose(a[0, :2], b[0, :2], atol=1e-12)

    def test_dropout_train_vs_eval(self) -> None:
        blk = EncoderBlock(D, H, DFF, dropout=0.5, random_state=0)
        x = Tensor(_x())
        assert not np.allclose(blk(x).data, blk(x).data)  # fresh masks each call
        blk.eval()
        np.testing.assert_array_equal(blk(x).data, blk(x).data)
        assert not blk.drop1.training and not blk.ffn.fc1.training
        blk.train()
        assert blk.drop1.training

    def test_gradient_check_input(self) -> None:
        blk = EncoderBlock(D, H, DFF, random_state=0)
        x = _x()
        R = rng.standard_normal((B, T, D))
        t = Tensor(x, requires_grad=True)
        (blk(t) * R).sum().backward()
        gradient_check(lambda a: float((blk(Tensor(a)) * R).sum().data), t.grad, x)

    def test_every_parameter_receives_a_gradient(self) -> None:
        blk = EncoderBlock(D, H, DFF, random_state=0)
        blk(Tensor(_x())).sum().backward()
        assert len(blk.parameters()) == 2 + 8 + 2 + 4  # ln1, mha, ln2, ffn
        assert all(p.grad is not None for p in blk.parameters())

    def test_same_seed_same_weights(self) -> None:
        a = EncoderBlock(D, H, DFF, random_state=3)
        b = EncoderBlock(D, H, DFF, random_state=3)
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            np.testing.assert_array_equal(p.data, q.data)


class TestDecoderBlock:
    def test_default_mask_is_causal(self) -> None:
        blk = DecoderBlock(D, H, DFF, cross_attention=False, random_state=0)
        x = _x()
        x2 = x.copy()
        x2[:, 3] = rng.standard_normal((B, D))  # change only the last position
        a, b = blk(Tensor(x)).data, blk(Tensor(x2)).data
        np.testing.assert_allclose(a[:, :3], b[:, :3], atol=1e-12)
        assert not np.allclose(a[:, 3], b[:, 3])

    def test_structure_with_cross_attention(self) -> None:
        blk = DecoderBlock(D, H, DFF, random_state=0)
        x, mem = _x(), _x(TK)
        mask = causal_mask(T, T)
        s = x + blk.self_attn(blk.ln1(Tensor(x)), mask=mask)[0].data
        c = s + blk.cross_attn(blk.ln_cross(Tensor(s)), Tensor(mem))[0].data
        ref = c + blk.ffn(blk.ln2(Tensor(c))).data
        np.testing.assert_allclose(blk(Tensor(x), Tensor(mem)).data, ref, atol=1e-12)

    def test_memory_padding_mask(self) -> None:
        blk = DecoderBlock(D, H, DFF, random_state=0)
        x, mem = Tensor(_x()), _x(TK)
        mem2 = mem.copy()
        mem2[0, 2:] = rng.standard_normal((TK - 2, D))
        mask = padding_mask([2, TK], TK)
        a = blk(x, Tensor(mem), memory_mask=mask).data
        b = blk(x, Tensor(mem2), memory_mask=mask).data
        np.testing.assert_allclose(a[0], b[0], atol=1e-12)

    def test_decoder_only_has_no_cross_parameters(self) -> None:
        with_cross = DecoderBlock(D, H, DFF, random_state=0)
        without = DecoderBlock(D, H, DFF, cross_attention=False, random_state=0)
        assert len(with_cross.parameters()) - len(without.parameters()) == 2 + 8
        without(Tensor(_x())).sum().backward()
        assert all(p.grad is not None for p in without.parameters())

    def test_gradients_reach_memory_and_every_parameter(self) -> None:
        blk = DecoderBlock(D, H, DFF, random_state=0)
        mem = Tensor(_x(TK), requires_grad=True)
        blk(Tensor(_x()), mem).sum().backward()
        assert mem.grad is not None
        assert all(p.grad is not None for p in blk.parameters())

    def test_gradient_check_memory(self) -> None:
        blk = DecoderBlock(D, H, DFF, random_state=0)
        x, mem = Tensor(_x()), _x(TK)
        R = rng.standard_normal((B, T, D))
        t = Tensor(mem, requires_grad=True)
        (blk(x, t) * R).sum().backward()
        gradient_check(lambda m: float((blk(x, Tensor(m)) * R).sum().data), t.grad, mem)

    def test_train_eval_reach_cross_sublayers(self) -> None:
        blk = DecoderBlock(D, H, DFF, dropout=0.1, random_state=0)
        blk.eval()
        assert not blk.drop_cross.training and not blk.cross_attn.training
        blk.train()
        assert blk.drop_cross.training

    def test_memory_validation(self) -> None:
        with pytest.raises(ValueError, match="memory is required"):
            DecoderBlock(D, H, DFF)(Tensor(_x()))
        with pytest.raises(ValueError, match="cross_attention=False"):
            DecoderBlock(D, H, DFF, cross_attention=False)(Tensor(_x()), Tensor(_x()))


def test_container_requires_children() -> None:
    from scratchgrad.transformer.block import _Container

    with pytest.raises(NotImplementedError):
        _Container().parameters()
