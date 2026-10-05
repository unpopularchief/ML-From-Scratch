"""PyTorch parity for scratchgrad.transformer and the layer-norm / GELU ops.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'``. Run with ``uv run pytest -m reference -q`` after
``uv sync --extra reference``. The blocks use the same formulas as
``nn.TransformerEncoderLayer`` / ``nn.TransformerDecoderLayer`` with
``norm_first=True`` (docs/derivations/transformer_block.md), so weights are copied
across and outputs and input gradients compared tightly.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.autograd import Tensor
from scratchgrad.autograd import functional as F
from scratchgrad.transformer import DecoderBlock, EncoderBlock

pytestmark = pytest.mark.reference

B, T, TK, D, H, DFF = 2, 5, 4, 8, 2, 16


def _t(torch, a):
    return torch.tensor(np.asarray(a), dtype=torch.float64)


def _copy_mha(torch, ours, theirs) -> None:
    with torch.no_grad():
        theirs.in_proj_weight.copy_(
            _t(
                torch,
                np.concatenate(
                    [lin.W.data.T for lin in (ours.w_q, ours.w_k, ours.w_v)]
                ),
            )
        )
        theirs.in_proj_bias.copy_(
            _t(
                torch,
                np.concatenate([lin.b.data for lin in (ours.w_q, ours.w_k, ours.w_v)]),
            )
        )
        theirs.out_proj.weight.copy_(_t(torch, ours.w_o.W.data.T))
        theirs.out_proj.bias.copy_(_t(torch, ours.w_o.b.data))


def _copy_ln(torch, ours, theirs) -> None:
    with torch.no_grad():
        theirs.weight.copy_(_t(torch, ours.gamma.data))
        theirs.bias.copy_(_t(torch, ours.beta.data))


def _copy_ffn(torch, ours, l1, l2) -> None:
    with torch.no_grad():
        l1.weight.copy_(_t(torch, ours.fc1.W.data.T))
        l1.bias.copy_(_t(torch, ours.fc1.b.data))
        l2.weight.copy_(_t(torch, ours.fc2.W.data.T))
        l2.bias.copy_(_t(torch, ours.fc2.b.data))


def _randomize_layer_norms(rng, *lns) -> None:
    for ln in lns:
        ln.gamma.data = rng.standard_normal(ln.gamma.data.shape) * 0.3 + 1.0
        ln.beta.data = rng.standard_normal(ln.beta.data.shape) * 0.3


def test_layer_norm_matches_torch_forward_and_backward():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x, g, b, dy = (
        rng.standard_normal((B, T, D)),
        rng.standard_normal(D),
        rng.standard_normal(D),
        rng.standard_normal((B, T, D)),
    )
    tx, tg, tb = (Tensor(a, requires_grad=True) for a in (x, g, b))
    out = F.layer_norm(tx, tg, tb)
    (out * dy).sum().backward()

    px, pg, pb = (_t(torch, a).requires_grad_() for a in (x, g, b))
    pout = torch.nn.functional.layer_norm(px, (D,), pg, pb, eps=1e-5)
    (pout * _t(torch, dy)).sum().backward()

    np.testing.assert_allclose(out.data, pout.detach().numpy(), atol=1e-10)
    for ours, theirs in ((tx, px), (tg, pg), (tb, pb)):
        np.testing.assert_allclose(ours.grad, theirs.grad.numpy(), atol=1e-10)


def test_gelu_matches_torch_tanh_forward_and_backward():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    x, dy = rng.standard_normal((B, T, D)) * 2, rng.standard_normal((B, T, D))
    tx = Tensor(x, requires_grad=True)
    out = F.gelu(tx)
    (out * dy).sum().backward()

    px = _t(torch, x).requires_grad_()
    pout = torch.nn.functional.gelu(px, approximate="tanh")
    (pout * _t(torch, dy)).sum().backward()

    np.testing.assert_allclose(out.data, pout.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(tx.grad, px.grad.numpy(), atol=1e-12)


def test_encoder_block_matches_torch_pre_ln_layer():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    blk = EncoderBlock(D, H, DFF, random_state=0)
    _randomize_layer_norms(rng, blk.ln1, blk.ln2)
    x, dy = rng.standard_normal((B, T, D)), rng.standard_normal((B, T, D))

    layer = torch.nn.TransformerEncoderLayer(
        D,
        H,
        DFF,
        dropout=0.0,
        activation=lambda t: torch.nn.functional.gelu(t, approximate="tanh"),
        batch_first=True,
        norm_first=True,
        dtype=torch.float64,
    )
    _copy_mha(torch, blk.attn, layer.self_attn)
    _copy_ln(torch, blk.ln1, layer.norm1)
    _copy_ln(torch, blk.ln2, layer.norm2)
    _copy_ffn(torch, blk.ffn, layer.linear1, layer.linear2)

    tx = Tensor(x, requires_grad=True)
    out = blk(tx)
    (out * dy).sum().backward()
    px = _t(torch, x).requires_grad_()
    pout = layer(px)
    (pout * _t(torch, dy)).sum().backward()

    np.testing.assert_allclose(out.data, pout.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(tx.grad, px.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(
        blk.ffn.fc1.W.grad, layer.linear1.weight.grad.numpy().T, atol=1e-10
    )


def test_decoder_block_matches_torch_pre_ln_layer():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    blk = DecoderBlock(D, H, DFF, random_state=0)
    _randomize_layer_norms(rng, blk.ln1, blk.ln_cross, blk.ln2)
    x, mem = rng.standard_normal((B, T, D)), rng.standard_normal((B, TK, D))
    dy = rng.standard_normal((B, T, D))

    layer = torch.nn.TransformerDecoderLayer(
        D,
        H,
        DFF,
        dropout=0.0,
        activation=lambda t: torch.nn.functional.gelu(t, approximate="tanh"),
        batch_first=True,
        norm_first=True,
        dtype=torch.float64,
    )
    _copy_mha(torch, blk.self_attn, layer.self_attn)
    _copy_mha(torch, blk.cross_attn, layer.multihead_attn)
    _copy_ln(torch, blk.ln1, layer.norm1)
    _copy_ln(torch, blk.ln_cross, layer.norm2)
    _copy_ln(torch, blk.ln2, layer.norm3)
    _copy_ffn(torch, blk.ffn, layer.linear1, layer.linear2)

    tx, tm = Tensor(x, requires_grad=True), Tensor(mem, requires_grad=True)
    out = blk(tx, tm)
    (out * dy).sum().backward()

    px, pm = _t(torch, x).requires_grad_(), _t(torch, mem).requires_grad_()
    causal = torch.nn.Transformer.generate_square_subsequent_mask(
        T, dtype=torch.float64
    )
    pout = layer(px, pm, tgt_mask=causal)
    (pout * _t(torch, dy)).sum().backward()

    np.testing.assert_allclose(out.data, pout.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(tx.grad, px.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(tm.grad, pm.grad.numpy(), atol=1e-10)
