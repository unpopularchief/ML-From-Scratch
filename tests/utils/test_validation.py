"""Tests for scratchgrad.utils.validation."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.exceptions import NotFittedError
from scratchgrad.utils.validation import (
    check_array,
    check_is_fitted,
    check_random_state,
    check_X_y,
)


class TestCheckRandomState:
    def test_int_seed_is_reproducible(self) -> None:
        a = check_random_state(0).standard_normal(5)
        b = check_random_state(0).standard_normal(5)
        np.testing.assert_array_equal(a, b)

    def test_generator_passed_through_unchanged(self) -> None:
        gen = np.random.default_rng(0)
        assert check_random_state(gen) is gen

    def test_none_returns_a_generator(self) -> None:
        assert isinstance(check_random_state(None), np.random.Generator)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError):
            check_random_state("not a seed")


class TestCheckArray:
    def test_coerces_to_float64(self) -> None:
        X = check_array([[1, 2], [3, 4]])
        assert X.dtype == np.float64
        np.testing.assert_array_equal(X, [[1.0, 2.0], [3.0, 4.0]])

    def test_rejects_1d_input(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            check_array([1.0, 2.0, 3.0])

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="0 samples"):
            check_array(np.empty((0, 3)))

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            check_array([[1.0, np.nan]])

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            check_array([[1.0, np.inf]])


class TestCheckXY:
    def test_valid_input_passes_through(self) -> None:
        X, y = check_X_y([[1.0, 2.0], [3.0, 4.0]], [0.0, 1.0])
        assert X.shape == (2, 2)
        assert y.shape == (2,)

    def test_rejects_2d_y(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            check_X_y([[1.0, 2.0]], [[0.0]])

    def test_rejects_mismatched_n_samples(self) -> None:
        with pytest.raises(ValueError, match="mismatched"):
            check_X_y([[1.0], [2.0], [3.0]], [0.0, 1.0])

    def test_rejects_nan_in_y(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            check_X_y([[1.0], [2.0]], [0.0, np.nan])


class TestCheckIsFitted:
    def test_raises_when_attribute_missing(self) -> None:
        class Dummy:
            pass

        with pytest.raises(NotFittedError):
            check_is_fitted(Dummy(), "coef_")

    def test_passes_when_attribute_present(self) -> None:
        class Dummy:
            coef_ = np.array([1.0])

        check_is_fitted(Dummy(), "coef_")  # should not raise

    def test_raises_if_any_of_several_attributes_missing(self) -> None:
        class Dummy:
            coef_ = np.array([1.0])

        with pytest.raises(NotFittedError):
            check_is_fitted(Dummy(), ["coef_", "intercept_"])
