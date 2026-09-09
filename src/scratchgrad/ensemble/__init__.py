"""Ensemble methods.

One file per algorithm. ``GradientBoosting`` and a
``RandomForestRegressor`` are likely later additions; only what is
implemented is exported.
"""

from scratchgrad.ensemble.adaboost import AdaBoostClassifier
from scratchgrad.ensemble.random_forest import RandomForestClassifier

__all__ = ["AdaBoostClassifier", "RandomForestClassifier"]
