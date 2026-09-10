"""Ensemble methods.

One file per algorithm. A ``RandomForestRegressor`` is a likely later
addition; only what is implemented is exported.
"""

from scratchgrad.ensemble.adaboost import AdaBoostClassifier
from scratchgrad.ensemble.gradient_boosting import GradientBoostingClassifier
from scratchgrad.ensemble.random_forest import RandomForestClassifier

__all__ = [
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    "RandomForestClassifier",
]
