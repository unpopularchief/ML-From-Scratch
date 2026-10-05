"""PyTorch parity for scratchgrad.attention.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'``. Run with ``uv run pytest -m reference -q`` after
``uv sync --extra reference``. Same formula on both sides
(docs/derivations/attention.md), so outputs and input gradients are compared
tightly; masks are applied the same way (blocked keys get ~zero weight).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.attention import (
    MultiHeadAttention,
    causal_mask,
    padding_mask,
    scaled_dot_product_attention,
)
from scratchgrad.autograd import Tensor

pytestmark = pytest.mark.reference

B, T, D_K, D_V = 2, 5, 4, 3


@pytest.mark.parametrize("kind", ["none", "causal", "padding", "both"])
def test_matches_torch_forward_and_backward(kind):
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    rng = np.random.default_rng(0)
    q, k, v = (
        rng.standard_normal((B, T, D_K)),
        rng.standard_normal((B, T, D_K)),
        rng.standard_normal((B, T, D_V)),
    )
    dO = rng.standard_normal((B, T, D_V))

    mask = None
    if kind in ("causal", "both"):
        mask = causal_mask(T, T)
    if kind in ("padding", "both"):
        pad = padding_mask([3, T], T)
        mask = pad if mask is None else mask & pad

    tq, tk, tv = (Tensor(a, requires_grad=True) for a in (q, k, v))
    out, _ = scaled_dot_product_attention(tq, tk, tv, mask)
    (out * dO).sum().backward()

    pq, pk, pv = (torch.tensor(a, requires_grad=True) for a in (q, k, v))
    pmask = None if mask is None else torch.tensor(np.broadcast_to(mask, (B, T, T)))
    pout = F.scaled_dot_product_attention(pq, pk, pv, attn_mask=pmask)
    (pout * torch.tensor(dO)).sum().backward()

    np.testing.assert_allclose(out.data, pout.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(tq.grad, pq.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(tk.grad, pk.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(tv.grad, pv.grad.numpy(), atol=1e-10)


@pytest.mark.parametrize("kind", ["none", "causal", "padding"])
def test_multi_head_matches_torch_forward_and_backward(kind):
    torch = pytest.importorskip("torch")
    d, h = 8, 2
    rng = np.random.default_rng(1)
    mha = MultiHeadAttention(d, h, random_state=0)
    x = rng.standard_normal((B, T, d))
    dO = rng.standard_normal((B, T, d))

    mask = None
    if kind == "causal":
        mask = causal_mask(T, T)
    elif kind == "padding":
        mask = padding_mask([3, T], T)

    tx = Tensor(x, requires_grad=True)
    out, weights = mha(tx, mask=mask)
    (out * dO).sum().backward()

    ref = torch.nn.MultiheadAttention(d, h, batch_first=True).double()
    with torch.no_grad():  # nn.Linear stores W as (out, in): ours is its transpose
        ref.in_proj_weight.copy_(
            torch.tensor(
                np.concatenate([m.W.data.T for m in (mha.w_q, mha.w_k, mha.w_v)])
            )
        )
        ref.in_proj_bias.copy_(
            torch.tensor(
                np.concatenate([m.b.data for m in (mha.w_q, mha.w_k, mha.w_v)])
            )
        )
        ref.out_proj.weight.copy_(torch.tensor(mha.w_o.W.data.T))
        ref.out_proj.bias.copy_(torch.tensor(mha.w_o.b.data))
    px = torch.tensor(x, requires_grad=True)
    # torch's boolean mask means "blocked", the opposite of ours
    pmask = None
    if mask is not None:
        pmask = torch.tensor(~np.broadcast_to(mask, (B, T, T))).repeat_interleave(h, 0)
    pout, pweights = ref(
        px, px, px, attn_mask=pmask, need_weights=True, average_attn_weights=False
    )
    (pout * torch.tensor(dO)).sum().backward()

    np.testing.assert_allclose(out.data, pout.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(weights.data, pweights.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(tx.grad, px.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(
        mha.w_o.W.grad, ref.out_proj.weight.grad.numpy().T, atol=1e-10
    )
    np.testing.assert_allclose(
        np.concatenate([m.W.grad.T for m in (mha.w_q, mha.w_k, mha.w_v)]),
        ref.in_proj_weight.grad.numpy(),
        atol=1e-10,
    )
