"""Classification metrics.

All functions here assume binary or multiclass labels encoded as integers
(or floats holding integer values) in ``y_true``/``y_pred``, shape
``(n_samples,)`` — the output of a classifier's ``predict``, not
``predict_proba``.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import IntArray, TargetVector


def _as_int_pair(
    y_true: TargetVector, y_pred: TargetVector
) -> tuple[IntArray, IntArray]:
    """Coerce both label arrays to int64 ndarrays. Internal helper."""
    return np.asarray(y_true).astype(np.int64), np.asarray(y_pred).astype(np.int64)


def accuracy_score(y_true: TargetVector, y_pred: TargetVector) -> float:
    r"""Fraction of predictions that exactly match the true label.

    .. math:: \mathrm{accuracy} = \frac{1}{n} \sum_i \mathbb{1}[y_i = \hat{y}_i]
    """
    y_true, y_pred = _as_int_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: TargetVector, y_pred: TargetVector) -> IntArray:
    """Confusion matrix over the classes observed in ``y_true``/``y_pred``.

    Entry ``[i, j]`` counts samples whose true label is class ``i`` and
    whose predicted label is class ``j`` — so the diagonal holds correct
    predictions and off-diagonal entries hold the specific mistakes made.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True class labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels.

    Returns
    -------
    ndarray of shape (n_classes, n_classes), dtype int64
        Classes are ordered by their sorted, unique label value.

    """
    y_true, y_pred = _as_int_pair(y_true, y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    index = {label: i for i, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[index[true_label], index[pred_label]] += 1
    return matrix


def _binary_counts(
    y_true: IntArray, y_pred: IntArray, positive_label: int
) -> tuple[int, int, int]:
    """Count true/false positives and false negatives for one class vs. the rest."""
    is_true_positive_class = y_true == positive_label
    is_pred_positive_class = y_pred == positive_label
    tp = int(np.sum(is_true_positive_class & is_pred_positive_class))
    fp = int(np.sum(~is_true_positive_class & is_pred_positive_class))
    fn = int(np.sum(is_true_positive_class & ~is_pred_positive_class))
    return tp, fp, fn


def precision_score(
    y_true: TargetVector, y_pred: TargetVector, positive_label: int = 1
) -> float:
    r"""Precision for ``positive_label``, :math:`\frac{TP}{TP + FP}`.

    Of the samples predicted as ``positive_label``, the fraction that
    actually are. Returns 0.0 (rather than raising) when the model never
    predicts the positive label, i.e. ``TP + FP == 0``.
    """
    y_true, y_pred = _as_int_pair(y_true, y_pred)
    tp, fp, _ = _binary_counts(y_true, y_pred, positive_label)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def recall_score(
    y_true: TargetVector, y_pred: TargetVector, positive_label: int = 1
) -> float:
    r"""Recall for ``positive_label``, :math:`\frac{TP}{TP + FN}`.

    Of the samples that actually are ``positive_label``, the fraction the
    model catches. Returns 0.0 when there are no true positive-label
    samples at all, i.e. ``TP + FN == 0``.
    """
    y_true, y_pred = _as_int_pair(y_true, y_pred)
    tp, _, fn = _binary_counts(y_true, y_pred, positive_label)
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0


def f1_score(
    y_true: TargetVector, y_pred: TargetVector, positive_label: int = 1
) -> float:
    r"""F1 score: harmonic mean of precision and recall.

    .. math:: F_1 = 2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
                                  {\mathrm{precision} + \mathrm{recall}}

    The harmonic mean (rather than the arithmetic mean) is used because it
    punishes a large gap between precision and recall — a classifier that
    scores 1.0 on one and 0.0 on the other gets F1 = 0, not 0.5.
    """
    precision = precision_score(y_true, y_pred, positive_label)
    recall = recall_score(y_true, y_pred, positive_label)
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))
