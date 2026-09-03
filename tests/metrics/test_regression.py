"""Tests for scratchgrad.metrics.regression — all analytic (tier 1) checks."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)


class TestMeanSquaredError:
    def test_hand_computed_example(self) -> None:
        # errors are [1, -1, 2] -> squared [1, 1, 4] -> mean 2.0
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 3.0, 1.0])
        assert mean_squared_error(y_true, y_pred) == pytest.approx(2.0)

    def test_zero_for_perfect_predictions(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y, y) == 0.0

    def test_symmetric_in_its_arguments(self) -> None:
        a, b = np.array([1.0, 2.0]), np.array([3.0, 1.0])
        assert mean_squared_error(a, b) == mean_squared_error(b, a)


class TestRootMeanSquaredError:
    def test_is_sqrt_of_mse(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 3.0, 1.0])
        assert root_mean_squared_error(y_true, y_pred) == pytest.approx(
            np.sqrt(mean_squared_error(y_true, y_pred))
        )


class TestMeanAbsoluteError:
    def test_hand_computed_example(self) -> None:
        # errors are [1, -1, 2] -> abs [1, 1, 2] -> mean 4/3
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 3.0, 1.0])
        assert mean_absolute_error(y_true, y_pred) == pytest.approx(4.0 / 3.0)

    def test_less_sensitive_to_outliers_than_mse(self) -> None:
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred_one_outlier = np.array([0.0, 0.0, 10.0])
        # MAE grows linearly with the outlier, MSE quadratically -> once
        # scaled to comparable units, MAE reports the smaller error.
        assert mean_absolute_error(y_true, y_pred_one_outlier) < mean_squared_error(
            y_true, y_pred_one_outlier
        )


class TestR2Score:
    def test_perfect_predictions_score_one(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_predicting_the_mean_scores_zero(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, np.mean(y_true))
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_worse_than_mean_scores_negative(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([10.0, -10.0, 20.0, -20.0])
        assert r2_score(y_true, y_pred) < 0.0

    def test_raises_when_y_true_is_constant(self) -> None:
        y_true = np.array([5.0, 5.0, 5.0])
        with pytest.raises(ValueError, match="undefined"):
            r2_score(y_true, np.array([1.0, 2.0, 3.0]))
