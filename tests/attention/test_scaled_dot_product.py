"""Scaled dot-product attention (docs/derivations/attention.md)."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.attention import (
    causal_mask,
    padding_mask,
    scaled_dot_product_attention,
)
from scratchgrad.autograd import Tensor
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)
B, T_Q, T_K, D_K, D_V = 2, 3, 4, 5, 6


def _inputs():
    return (
        rng.standard_normal((B, T_Q, D_K)),
        rng.standard_normal((B, T_K, D_K)),
        rng.standard_normal((B, T_K, D_V)),
    )


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TestForward:
    def test_matches_numpy_formula(self):
        q, k, v = _inputs()
        out, w = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v))
        expected_w = _softmax(q @ k.transpose(0, 2, 1) / np.sqrt(D_K))
        np.testing.assert_allclose(w.data, expected_w, atol=1e-12)
        np.testing.assert_allclose(out.data, expected_w @ v, atol=1e-12)

    def test_weights_are_row_stochastic(self):
        q, k, v = _inputs()
        _, w = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v))
        np.testing.assert_allclose(w.data.sum(axis=-1), 1.0)

    def test_two_dim_inputs(self):
        q, k, v = _inputs()
        out, w = scaled_dot_product_attention(Tensor(q[0]), Tensor(k[0]), Tensor(v[0]))
        assert out.shape == (T_Q, D_V) and w.shape == (T_Q, T_K)

    def test_scale_keeps_logit_variance_near_one(self):
        # Var(q.k) = d_k for unit-variance entries, so the 1/sqrt(d_k) scale gives ~1.
        d = 256
        q, k = rng.standard_normal((2, 2000, d))
        scores = (q * k).sum(axis=-1) / np.sqrt(d)
        assert scores.var() == pytest.approx(1.0, abs=0.15)

    def test_identical_keys_give_uniform_weights(self):
        q = rng.standard_normal((1, T_Q, D_K))
        k = np.tile(rng.standard_normal((1, 1, D_K)), (1, T_K, 1))
        v = rng.standard_normal((1, T_K, D_V))
        _, w = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v))
        np.testing.assert_allclose(w.data, 1.0 / T_K)


class TestMasks:
    def test_causal_blocks_future(self):
        q, k, v = _inputs()
        k, v = k[:, :T_Q], v[:, :T_Q]
        _, w = scaled_dot_product_attention(
            Tensor(q), Tensor(k), Tensor(v), causal_mask(T_Q, T_Q)
        )
        np.testing.assert_array_equal(np.triu(w.data, k=1), 0.0)
        np.testing.assert_allclose(w.data.sum(axis=-1), 1.0)
        np.testing.assert_allclose(w.data[:, 0, 0], 1.0)  # first query sees itself only

    def test_causal_output_ignores_future_values(self):
        q, k, v = _inputs()
        k, v = k[:, :T_Q], v[:, :T_Q]
        mask = causal_mask(T_Q, T_Q)
        out, _ = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask)
        v2 = v.copy()
        v2[:, -1] += 100.0  # perturb the last position only
        out2, _ = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v2), mask)
        np.testing.assert_allclose(out.data[:, :-1], out2.data[:, :-1])

    def test_padding_gives_zero_weight_to_pad_keys(self):
        q, k, v = _inputs()
        lengths = [2, 4]
        _, w = scaled_dot_product_attention(
            Tensor(q), Tensor(k), Tensor(v), padding_mask(lengths, T_K)
        )
        np.testing.assert_array_equal(w.data[0, :, 2:], 0.0)
        np.testing.assert_allclose(w.data.sum(axis=-1), 1.0)

    def test_padding_equals_truncating_the_sequence(self):
        q, k, v = _inputs()
        out, _ = scaled_dot_product_attention(
            Tensor(q), Tensor(k), Tensor(v), padding_mask([2, 4], T_K)
        )
        short, _ = scaled_dot_product_attention(
            Tensor(q[:1]), Tensor(k[:1, :2]), Tensor(v[:1, :2])
        )
        np.testing.assert_allclose(out.data[0], short.data[0], atol=1e-12)

    def test_combined_masks_broadcast(self):
        q, k, v = _inputs()
        k, v = k[:, :T_Q], v[:, :T_Q]
        mask = causal_mask(T_Q, T_Q) & padding_mask([2, 3], T_Q)
        _, w = scaled_dot_product_attention(Tensor(q), Tensor(k), Tensor(v), mask)
        np.testing.assert_array_equal(np.triu(w.data, k=1), 0.0)
        np.testing.assert_array_equal(w.data[0, :, 2:], 0.0)

    def test_fully_masked_row_stays_finite(self):
        q, k, v = _inputs()
        out, w = scaled_dot_product_attention(
            Tensor(q), Tensor(k), Tensor(v), padding_mask([0, 4], T_K)
        )
        assert np.isfinite(out.data).all() and np.isfinite(w.data).all()


class TestGradients:
    @pytest.mark.parametrize("masked", [False, True])
    def test_gradient_check_each_input(self, masked):
        q, k, v = _inputs()
        mask = padding_mask([2, 4], T_K) if masked else None
        R = rng.standard_normal((B, T_Q, D_V))

        def loss(q_, k_, v_):
            out, _ = scaled_dot_product_attention(q_, k_, v_, mask)
            return (out * R).sum()

        tq, tk, tv = (Tensor(a, requires_grad=True) for a in (q, k, v))
        loss(tq, tk, tv).backward()
        gradient_check(
            lambda x: float(loss(Tensor(x), Tensor(k), Tensor(v)).data), tq.grad, q
        )
        gradient_check(
            lambda x: float(loss(Tensor(q), Tensor(x), Tensor(v)).data), tk.grad, k
        )
        gradient_check(
            lambda x: float(loss(Tensor(q), Tensor(k), Tensor(x)).data), tv.grad, v
        )

    def test_autograd_matches_hand_derived_gradients(self):
        """dV=A^T dO, dA=dO V^T, dS=A*(dA-sum(dA*A)), dQ=dS K/s, dK=dS^T Q/s."""
        q, k, v = _inputs()
        mask = padding_mask([3, 4], T_K)
        dO = rng.standard_normal((B, T_Q, D_V))
        tq, tk, tv = (Tensor(a, requires_grad=True) for a in (q, k, v))
        out, w = scaled_dot_product_attention(tq, tk, tv, mask)
        (out * dO).sum().backward()

        A, s = w.data, np.sqrt(D_K)
        dV = A.transpose(0, 2, 1) @ dO
        dA = dO @ v.transpose(0, 2, 1)
        dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        np.testing.assert_allclose(tv.grad, dV, atol=1e-10)
        np.testing.assert_allclose(tq.grad, dS @ k / s, atol=1e-10)
        np.testing.assert_allclose(tk.grad, dS.transpose(0, 2, 1) @ q / s, atol=1e-10)

    def test_masked_positions_get_no_key_or_value_gradient(self):
        q, k, v = _inputs()
        tk, tv = Tensor(k, requires_grad=True), Tensor(v, requires_grad=True)
        out, _ = scaled_dot_product_attention(
            Tensor(q), tk, tv, padding_mask([2, 4], T_K)
        )
        out.sum().backward()
        np.testing.assert_array_equal(tv.grad[0, 2:], 0.0)
        np.testing.assert_array_equal(tk.grad[0, 2:], 0.0)


class TestValidation:
    def test_key_dim_mismatch(self):
        with pytest.raises(ValueError, match="same last dim"):
            scaled_dot_product_attention(
                Tensor(np.ones((2, 3))),
                Tensor(np.ones((2, 4))),
                Tensor(np.ones((2, 4))),
            )

    def test_key_value_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            scaled_dot_product_attention(
                Tensor(np.ones((2, 3))),
                Tensor(np.ones((2, 3))),
                Tensor(np.ones((5, 3))),
            )
