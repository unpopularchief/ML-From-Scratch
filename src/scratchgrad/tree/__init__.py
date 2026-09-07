"""Decision trees.

One file per algorithm. ``DecisionTreeRegressor`` is a likely later
addition (MSE criterion, mean-value leaves); only what is implemented is
exported.
"""

from scratchgrad.tree.decision_tree import DecisionTreeClassifier

__all__ = ["DecisionTreeClassifier"]
