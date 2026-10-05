"""Multi-head attention (docs/derivations/multi_head_positional.md section 1)."""

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
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)
B, T_Q, T_K, D, H = 2, 3, 4, 8, 2


def _mha():
    return MultiHeadAttention(D, H, random_state=0)


def _reference(mha, xq, xk, xv, mask=None):
    """Per-head Python loop: slice the projections' columns head by head."""
    dh = D // H
    heads = []
    for i in range(H):
        cols = slice(i * dh, (i + 1) * dh)
        q = xq @ mha.w_q.W.data[:, cols] + mha.w_q.b.data[cols]
        k = xk @ mha.w_k.W.data[:, cols] + mha.w_k.b.data[cols]
        v = xv @ mha.w_v.W.data[:, cols] + mha.w_v.b.data[cols]
        out, _ = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask)
        heads.append(out.data)
    return np.concatenate(heads, axis=-1) @ mha.w_o.W.data + mha.w_o.b.data


class TestForward:
    def test_shapes(self):
        out, w = _mha()(Tensor(rng.standard_normal((B, T_Q, D))))
        assert out.shape == (B, T_Q, D) and w.shape == (B, H, T_Q, T_Q)

    def test_cross_attention_shapes(self):
        q = Tensor(rng.standard_normal((B, T_Q, D)))
        kv = Tensor(rng.standard_normal((B, T_K, D)))
        out, w = _mha()(q, kv)
        assert out.shape == (B, T_Q, D) and w.shape == (B, H, T_Q, T_K)

    def test_matches_per_head_loop(self):
        mha = _mha()
        xq = rng.standard_normal((B, T_Q, D))
        xk = rng.standard_normal((B, T_K, D))
        xv = rng.standard_normal((B, T_K, D))
        out, _ = mha(Tensor(xq), Tensor(xk), Tensor(xv))
        np.testing.assert_allclose(out.data, _reference(mha, xq, xk, xv), atol=1e-12)

    def test_single_head_equals_sdpa_with_projections(self):
        mha = MultiHeadAttention(D, 1, random_state=1)
        x = rng.standard_normal((B, T_Q, D))
        out, w = mha(Tensor(x))
        q, k, v = (x @ lin.W.data + lin.b.data for lin in (mha.w_q, mha.w_k, mha.w_v))
        ref, ref_w = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v))
        np.testing.assert_allclose(w.data[:, 0], ref_w.data, atol=1e-12)
        np.testing.assert_allclose(
            out.data, ref.data @ mha.w_o.W.data + mha.w_o.b.data, atol=1e-12
        )

    def test_weights_are_row_stochastic(self):
        _, w = _mha()(Tensor(rng.standard_normal((B, T_Q, D))))
        np.testing.assert_allclose(w.data.sum(axis=-1), 1.0)

    def test_same_seed_same_weights(self):
        a, b = _mha(), _mha()
        for p, q in zip(a.parameters(), b.parameters(), strict=True):
            np.testing.assert_array_equal(p.data, q.data)

    def test_parameters(self):
        params = _mha().parameters()
        assert len(params) == 8
        assert sum(p.data.size for p in params) == 4 * D * D + 4 * D


class TestMasks:
    def test_causal_blocks_future_in_every_head(self):
        x = Tensor(rng.standard_normal((B, T_Q, D)))
        _, w = _mha()(x, mask=causal_mask(T_Q, T_Q))
        np.testing.assert_array_equal(np.triu(w.data, k=1), 0.0)

    def test_padding_mask_gets_a_head_axis(self):
        x = Tensor(rng.standard_normal((B, T_K, D)))
        _, w = _mha()(x, mask=padding_mask([2, 4], T_K))
        np.testing.assert_array_equal(w.data[0, :, :, 2:], 0.0)
        assert (w.data[1] > 0).all()

    def test_combined_mask_matches_reference(self):
        mha = _mha()
        x = rng.standard_normal((B, T_K, D))
        mask = causal_mask(T_K, T_K) & padding_mask([2, 4], T_K)
        out, _ = mha(Tensor(x), mask=mask)
        np.testing.assert_allclose(out.data, _reference(mha, x, x, x, mask), atol=1e-12)

    def test_per_head_four_dim_mask_is_passed_through(self):
        x = Tensor(rng.standard_normal((B, T_Q, D)))
        mask = np.ones((B, H, T_Q, T_Q), dtype=bool)
        mask[:, 0] = causal_mask(T_Q, T_Q)  # head 0 causal, head 1 unmasked
        _, w = _mha()(x, mask=mask)
        np.testing.assert_array_equal(np.triu(w.data[:, 0], k=1), 0.0)
        assert (np.triu(w.data[:, 1], k=1) > 0).any()


class TestGradients:
    def test_gradient_check_inputs(self):
        mha = _mha()
        xq = rng.standard_normal((B, T_Q, D))
        xkv = rng.standard_normal((B, T_K, D))
        R = rng.standard_normal((B, T_Q, D))
        mask = padding_mask([2, 4], T_K)

        def loss(q, kv):
            return (mha(q, kv, mask=mask)[0] * R).sum()

        tq, tkv = Tensor(xq, requires_grad=True), Tensor(xkv, requires_grad=True)
        loss(tq, tkv).backward()
        gradient_check(lambda x: float(loss(Tensor(x), Tensor(xkv)).data), tq.grad, xq)
        gradient_check(lambda x: float(loss(Tensor(xq), Tensor(x)).data), tkv.grad, xkv)

    @pytest.mark.parametrize("which", ["w_q", "w_k", "w_v", "w_o"])
    def test_gradient_check_weights(self, which):
        mha = _mha()
        x = Tensor(rng.standard_normal((B, T_Q, D)))
        R = rng.standard_normal((B, T_Q, D))
        W = getattr(mha, which).W

        def loss():
            return (mha(x)[0] * R).sum()

        loss().backward()
        analytic = W.grad.copy()
        original = W.data.copy()

        def f(w):
            W.data = w
            return float(loss().data)

        gradient_check(f, analytic, original)
        W.data = original

    def test_every_parameter_receives_a_gradient(self):
        mha = _mha()
        mha(Tensor(rng.standard_normal((B, T_Q, D))))[0].sum().backward()
        assert all(p.grad is not None for p in mha.parameters())


class TestValidation:
    def test_indivisible_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadAttention(10, 3)

    def test_nonpositive_sizes(self):
        with pytest.raises(ValueError, match="positive"):
            MultiHeadAttention(0, 1)

    def test_wrong_input_shape(self):
        with pytest.raises(ValueError, match="must have shape"):
            _mha()(Tensor(np.ones((B, T_Q, D + 1))))
        with pytest.raises(ValueError, match="must have shape"):
            _mha()(Tensor(np.ones((T_Q, D))))
