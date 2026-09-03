"""Evaluation metrics for regression, classification, and pairwise distance."""

from scratchgrad.metrics.classification import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from scratchgrad.metrics.pairwise import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
)
from scratchgrad.metrics.regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "cosine_similarity",
    "euclidean_distance",
    "f1_score",
    "manhattan_distance",
    "mean_absolute_error",
    "mean_squared_error",
    "precision_score",
    "r2_score",
    "recall_score",
    "root_mean_squared_error",
]
