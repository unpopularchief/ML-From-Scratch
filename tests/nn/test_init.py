"""Tests for scratchgrad.nn.init (zeros, xavier_uniform, he_normal).

Tiers (plan.md section 3, docs/derivations/nn.md section 6): analytic
shape/value checks, statistical mean/variance checks against the
closed-form target, and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn.init import he_normal, xavier_uniform, zeros


class TestZeros:
    def test_returns_all_zero_float64_array_of_requested_shape(self) -> None:
        out = zeros((3, 4))
        assert out.shape == (3, 4)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, np.zeros((3, 4)))


class TestXavierUniform:
    def test_shape_and_dtype(self, rng: np.random.Generator) -> None:
        out = xavier_uniform(5, 3, rng)
        assert out.shape == (5, 3)
        assert out.dtype == np.float64

    def test_bounded_by_a(self, rng: np.random.Generator) -> None:
        fan_in, fan_out = 5, 3
        a = np.sqrt(6.0 / (fan_in + fan_out))
        out = xavier_uniform(fan_in, fan_out, rng)
        assert np.all(np.abs(out) <= a)

    def test_empirical_variance_matches_uniform_closed_form(self) -> None:
        # Var(U(-a,a)) = a^2/3. Large draw -> tight statistical tolerance.
        fan_in, fan_out = 50, 50
        rng = np.random.default_rng(0)
        a = np.sqrt(6.0 / (fan_in + fan_out))
        expected_var = a**2 / 3.0
        out = xavier_uniform(fan_in, fan_out, rng)
        assert out.var() == pytest.approx(expected_var, rel=0.1)


class TestHeNormal:
    def test_shape_and_dtype(self, rng: np.random.Generator) -> None:
        out = he_normal(5, 3, rng)
        assert out.shape == (5, 3)
        assert out.dtype == np.float64

    def test_empirical_mean_and_variance_match_closed_form(self) -> None:
        fan_in, fan_out = 50, 50
        rng = np.random.default_rng(0)
        out = he_normal(fan_in, fan_out, rng)
        assert abs(out.mean()) < 0.05
        assert out.var() == pytest.approx(2.0 / fan_in, rel=0.1)


class TestDeterminism:
    def test_same_seed_gives_byte_identical_draws(self) -> None:
        a = xavier_uniform(4, 4, np.random.default_rng(0))
        b = xavier_uniform(4, 4, np.random.default_rng(0))
        np.testing.assert_array_equal(a, b)
