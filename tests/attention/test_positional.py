"""Positional encodings (docs/derivations/multi_head_positional.md section 2)."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.attention import (
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
    sinusoidal_positional_encoding,
)
from scratchgrad.autograd import Tensor
from tests.helpers.gradcheck import gradient_check

rng = np.random.default_rng(0)


class TestSinusoidalTable:
    def test_known_values(self):
        pe = sinusoidal_positional_encoding(3, 4)
        omega = [1.0, 10000.0 ** (-2 / 4)]
        expected = [
            [np.sin(p * w) if j == 0 else np.cos(p * w) for w in omega for j in (0, 1)]
            for p in range(3)
        ]
        np.testing.assert_allclose(pe, expected, atol=1e-14)

    def test_position_zero_is_sin0_cos0(self):
        np.testing.assert_array_equal(
            sinusoidal_positional_encoding(1, 6)[0], [0, 1, 0, 1, 0, 1]
        )

    def test_entries_bounded(self):
        assert np.abs(sinusoidal_positional_encoding(200, 16)).max() <= 1.0

    def test_rows_are_distinct(self):
        pe = sinusoidal_positional_encoding(50, 16)
        assert len({tuple(r.round(9)) for r in pe}) == 50

    def test_shift_is_a_rotation_per_frequency_pair(self):
        """PE(p+k) = R(k w_i) PE(p) on each (sin, cos) pair, independent of p."""
        d, k = 8, 7
        pe = sinusoidal_positional_encoding(40, d)
        omega = 10000.0 ** (-np.arange(0, d, 2) / d)
        for p in (0, 3, 20):
            for i, w in enumerate(omega):
                s, c = pe[p, 2 * i], pe[p, 2 * i + 1]
                np.testing.assert_allclose(
                    pe[p + k, 2 * i],
                    s * np.cos(k * w) + c * np.sin(k * w),
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    pe[p + k, 2 * i + 1],
                    c * np.cos(k * w) - s * np.sin(k * w),
                    atol=1e-12,
                )

    @pytest.mark.parametrize("d_model", [0, 3, -2])
    def test_rejects_bad_d_model(self, d_model):
        with pytest.raises(ValueError, match="even"):
            sinusoidal_positional_encoding(4, d_model)

    def test_rejects_bad_max_len(self):
        with pytest.raises(ValueError, match="max_len"):
            sinusoidal_positional_encoding(0, 4)


class TestSinusoidalLayer:
    def test_adds_the_table(self):
        x = rng.standard_normal((2, 5, 6))
        out = SinusoidalPositionalEncoding(6, max_len=10)(Tensor(x))
        np.testing.assert_allclose(
            out.data, x + sinusoidal_positional_encoding(10, 6)[:5], atol=1e-14
        )

    def test_no_parameters_and_gradient_passes_through(self):
        layer = SinusoidalPositionalEncoding(4, max_len=8)
        assert layer.parameters() == []
        x = Tensor(rng.standard_normal((2, 3, 4)), requires_grad=True)
        layer(x).sum().backward()
        np.testing.assert_array_equal(x.grad, np.ones((2, 3, 4)))

    def test_too_long_sequence(self):
        with pytest.raises(ValueError, match="exceeds max_len"):
            SinusoidalPositionalEncoding(4, max_len=2)(Tensor(np.zeros((1, 3, 4))))

    def test_wrong_feature_dim(self):
        with pytest.raises(ValueError, match="must have shape"):
            SinusoidalPositionalEncoding(4, max_len=8)(Tensor(np.zeros((1, 3, 5))))


class TestLearned:
    def test_adds_first_t_rows(self):
        layer = LearnedPositionalEncoding(6, max_len=10, random_state=0)
        x = rng.standard_normal((2, 5, 6))
        np.testing.assert_allclose(
            layer(Tensor(x)).data, x + layer.table.data[:5], atol=1e-14
        )

    def test_same_seed_same_table(self):
        a = LearnedPositionalEncoding(4, 8, random_state=3)
        b = LearnedPositionalEncoding(4, 8, random_state=3)
        np.testing.assert_array_equal(a.table.data, b.table.data)

    def test_gradient_is_batch_sum_in_first_t_rows_and_zero_elsewhere(self):
        layer = LearnedPositionalEncoding(4, max_len=8, random_state=0)
        dO = rng.standard_normal((2, 3, 4))
        (layer(Tensor(np.zeros((2, 3, 4)))) * dO).sum().backward()
        np.testing.assert_allclose(layer.table.grad[:3], dO.sum(axis=0), atol=1e-14)
        np.testing.assert_array_equal(layer.table.grad[3:], 0.0)

    def test_gradient_check_table(self):
        layer = LearnedPositionalEncoding(4, max_len=6, random_state=0)
        x = Tensor(rng.standard_normal((2, 3, 4)))
        R = rng.standard_normal((2, 3, 4))
        (layer(x) * R).sum().backward()
        analytic, original = layer.table.grad.copy(), layer.table.data.copy()

        def f(t):
            layer.table.data = t
            return float((layer(x) * R).sum().data)

        gradient_check(f, analytic, original)

    def test_parameters(self):
        layer = LearnedPositionalEncoding(4, 8)
        assert layer.parameters() == [layer.table]

    def test_too_long_sequence(self):
        with pytest.raises(ValueError, match="exceeds max_len"):
            LearnedPositionalEncoding(4, max_len=2)(Tensor(np.zeros((1, 3, 4))))

    @pytest.mark.parametrize(
        "kwargs", [{"d_model": 0, "max_len": 4}, {"d_model": 4, "max_len": 0}]
    )
    def test_rejects_nonpositive_sizes(self, kwargs):
        with pytest.raises(ValueError, match="positive"):
            LearnedPositionalEncoding(**kwargs)
