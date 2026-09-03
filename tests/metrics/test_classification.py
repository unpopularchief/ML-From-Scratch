"""Tests for scratchgrad.metrics.classification — all analytic (tier 1) checks."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class TestAccuracyScore:
    def test_hand_computed_example(self) -> None:
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        assert accuracy_score(y_true, y_pred) == pytest.approx(0.75)

    def test_perfect_predictions_score_one(self) -> None:
        y = np.array([0, 1, 2, 1])
        assert accuracy_score(y, y) == 1.0

    def test_all_wrong_scores_zero(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert accuracy_score(y_true, y_pred) == 0.0


class TestConfusionMatrix:
    def test_hand_computed_binary_example(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        # true=0: predicted 0 once, predicted 1 once -> row [1, 1]
        # true=1: predicted 0 never, predicted 1 twice -> row [0, 2]
        expected = np.array([[1, 1], [0, 2]])
        np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)

    def test_diagonal_sums_to_number_correct(self) -> None:
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 2])
        matrix = confusion_matrix(y_true, y_pred)
        assert np.trace(matrix) == np.sum(y_true == y_pred)

    def test_total_count_equals_n_samples(self) -> None:
        y_true = np.array([0, 1, 2, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 2])
        assert np.sum(confusion_matrix(y_true, y_pred)) == len(y_true)


class TestPrecisionRecallF1:
    # Worked example: 5 samples, positive_label=1.
    #   y_true = [1, 1, 0, 0, 1]
    #   y_pred = [1, 0, 0, 1, 1]
    # TP=2 (idx 0, 4), FP=1 (idx 3), FN=1 (idx 1)
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1, 1])

    def test_precision_hand_computed(self) -> None:
        # precision = TP / (TP + FP) = 2 / 3
        assert precision_score(self.y_true, self.y_pred) == pytest.approx(2.0 / 3.0)

    def test_recall_hand_computed(self) -> None:
        # recall = TP / (TP + FN) = 2 / 3
        assert recall_score(self.y_true, self.y_pred) == pytest.approx(2.0 / 3.0)

    def test_f1_hand_computed(self) -> None:
        # precision == recall == 2/3 here, so F1 == 2/3 too (harmonic mean
        # of two equal numbers equals that number).
        assert f1_score(self.y_true, self.y_pred) == pytest.approx(2.0 / 3.0)

    def test_precision_zero_when_no_positive_predictions(self) -> None:
        y_true = np.array([1, 1, 0])
        y_pred = np.array([0, 0, 0])
        assert precision_score(y_true, y_pred) == 0.0

    def test_recall_zero_when_no_actual_positives(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 0, 1])
        assert recall_score(y_true, y_pred) == 0.0

    def test_f1_zero_when_precision_and_recall_both_zero(self) -> None:
        y_true = np.array([1, 1])
        y_pred = np.array([0, 0])
        assert f1_score(y_true, y_pred) == 0.0

    def test_perfect_predictions_score_one_on_all_three(self) -> None:
        y = np.array([1, 0, 1, 1, 0])
        assert precision_score(y, y) == 1.0
        assert recall_score(y, y) == 1.0
        assert f1_score(y, y) == 1.0
