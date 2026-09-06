"""Nearest-neighbour methods.

One file per algorithm. ``KNeighborsRegressor`` is a likely later addition;
only what is implemented is exported.
"""

from scratchgrad.neighbors.knn import KNeighborsClassifier

__all__ = ["KNeighborsClassifier"]
