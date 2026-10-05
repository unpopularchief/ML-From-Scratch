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
