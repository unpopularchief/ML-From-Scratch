"""Tests for scratchgrad.preprocessing.split.train_test_split."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.preprocessing.split import train_test_split


class TestTrainTestSplit:
    def test_split_sizes_match_test_size(self) -> None:
        X = np.arange(20).reshape(20, 1).astype(np.float64)
        y = np.arange(20).astype(np.float64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=0
        )
        assert X_test.shape[0] == 5
        assert X_train.shape[0] == 15
        assert y_test.shape[0] == 5
        assert y_train.shape[0] == 15

    def test_every_original_sample_appears_exactly_once(self) -> None:
        X = np.arange(10).reshape(10, 1).astype(np.float64)
        y = np.arange(10).astype(np.float64)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0
        )
        all_indices = np.concatenate([X_train.ravel(), X_test.ravel()])
        np.testing.assert_array_equal(np.sort(all_indices), np.arange(10))

    def test_X_and_y_rows_stay_paired_after_shuffling(self) -> None:
        # y is a direct function of X here, so if the split ever shuffled
        # X and y independently this identity would break.
        X = np.arange(20).reshape(20, 1).astype(np.float64)
        y = X.ravel() * 10.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        np.testing.assert_array_equal(y_train, X_train.ravel() * 10.0)
        np.testing.assert_array_equal(y_test, X_test.ravel() * 10.0)

    def test_same_seed_gives_same_split(self) -> None:
        X = np.arange(20).reshape(20, 1).astype(np.float64)
        y = np.arange(20).astype(np.float64)
        split_a = train_test_split(X, y, random_state=42)
        split_b = train_test_split(X, y, random_state=42)
        for a, b in zip(split_a, split_b, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_invalid_test_size_raises(self) -> None:
        X, y = np.zeros((10, 1)), np.zeros(10)
        with pytest.raises(ValueError, match="test_size"):
            train_test_split(X, y, test_size=1.5)

    def test_test_size_too_small_for_n_samples_raises(self) -> None:
        X, y = np.zeros((2, 1)), np.zeros(2)
        with pytest.raises(ValueError, match="empty"):
            train_test_split(X, y, test_size=0.1)
