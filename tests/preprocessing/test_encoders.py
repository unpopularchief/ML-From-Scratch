"""Tests for scratchgrad.preprocessing.encoders.OneHotEncoder."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.exceptions import NotFittedError
from scratchgrad.preprocessing.encoders import OneHotEncoder


class TestOneHotEncoder:
    def test_hand_computed_example(self) -> None:
        y = np.array([0, 2, 1])
        one_hot = OneHotEncoder().fit_transform(y)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        np.testing.assert_array_equal(one_hot, expected)

    def test_each_row_sums_to_one(self) -> None:
        y = np.array([1, 3, 3, 1, 5])
        one_hot = OneHotEncoder().fit_transform(y)
        np.testing.assert_array_equal(np.sum(one_hot, axis=1), np.ones(len(y)))

    def test_categories_are_sorted(self) -> None:
        encoder = OneHotEncoder().fit(np.array([5, 1, 3]))
        np.testing.assert_array_equal(encoder.categories_, [1, 3, 5])

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(NotFittedError):
            OneHotEncoder().transform(np.array([0, 1]))

    def test_unseen_category_at_transform_raises(self) -> None:
        encoder = OneHotEncoder().fit(np.array([0, 1]))
        with pytest.raises(ValueError, match="Unseen categories"):
            encoder.transform(np.array([0, 2]))

    def test_inverse_transform_recovers_original_labels(self) -> None:
        y = np.array([2, 0, 1, 2])
        encoder = OneHotEncoder()
        one_hot = encoder.fit_transform(y)
        np.testing.assert_array_equal(encoder.inverse_transform(one_hot), y)
