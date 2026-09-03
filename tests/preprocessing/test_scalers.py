"""Tests for scratchgrad.preprocessing.scalers."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.exceptions import NotFittedError
from scratchgrad.preprocessing.scalers import MinMaxScaler, StandardScaler


class TestStandardScaler:
    def test_transform_gives_zero_mean_unit_variance(self) -> None:
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        Xt = StandardScaler().fit_transform(X)
        assert np.mean(Xt) == pytest.approx(0.0, abs=1e-10)
        assert np.std(Xt, ddof=0) == pytest.approx(1.0)

    def test_constant_feature_does_not_divide_by_zero(self) -> None:
        X = np.array([[5.0], [5.0], [5.0]])
        Xt = StandardScaler().fit_transform(X)
        assert np.all(np.isfinite(Xt))
        np.testing.assert_allclose(Xt, 0.0)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            StandardScaler().transform(np.array([[1.0]]))

    def test_inverse_transform_undoes_transform(self) -> None:
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        scaler = StandardScaler().fit(X)
        recovered = scaler.inverse_transform(scaler.transform(X))
        np.testing.assert_allclose(recovered, X, atol=1e-10)

    def test_test_set_uses_train_statistics_not_its_own(self) -> None:
        # The whole point of fit/transform being separate: transforming
        # new data must not silently refit on it.
        X_train = np.array([[0.0], [10.0]])
        X_test = np.array([[100.0]])
        scaler = StandardScaler().fit(X_train)
        # train mean=5, std=5 -> (100 - 5) / 5 = 19, not 0 (which fitting
        # fresh on X_test alone would give, since std of one point is 0).
        assert scaler.transform(X_test)[0, 0] == pytest.approx(19.0)


class TestMinMaxScaler:
    def test_default_range_is_zero_to_one(self) -> None:
        X = np.array([[0.0], [5.0], [10.0]])
        Xt = MinMaxScaler().fit_transform(X)
        np.testing.assert_allclose(Xt, [[0.0], [0.5], [1.0]])

    def test_custom_feature_range(self) -> None:
        X = np.array([[0.0], [10.0]])
        Xt = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
        np.testing.assert_allclose(Xt, [[-1.0], [1.0]])

    def test_constant_feature_maps_to_range_low(self) -> None:
        X = np.array([[5.0], [5.0]])
        Xt = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
        np.testing.assert_allclose(Xt, [[-1.0], [-1.0]])

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            MinMaxScaler().transform(np.array([[1.0]]))

    def test_get_params_reports_feature_range(self) -> None:
        scaler = MinMaxScaler(feature_range=(-2.0, 2.0))
        assert scaler.get_params() == {"feature_range": (-2.0, 2.0)}
