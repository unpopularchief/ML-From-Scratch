"""ML From Scratch: machine learning and deep learning built from first principles.

The public API is curated here as each algorithm lands (see ROADMAP.md for
progress); nothing is exported from this file until it exists.
"""

from scratchgrad.cluster import DBSCAN, KMeans
from scratchgrad.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from scratchgrad.linear import Lasso, LinearRegression, LogisticRegression, Ridge
from scratchgrad.naive_bayes import GaussianNB
from scratchgrad.neighbors import KNeighborsClassifier
from scratchgrad.tree import DecisionTreeClassifier, DecisionTreeRegressor

__version__ = "0.0.1"

__all__ = [
    "AdaBoostClassifier",
    "DBSCAN",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "GaussianNB",
    "GradientBoostingClassifier",
    "KMeans",
    "KNeighborsClassifier",
    "Lasso",
    "LinearRegression",
    "LogisticRegression",
    "RandomForestClassifier",
    "Ridge",
]
