"""Decision trees.

One file per algorithm: ``DecisionTreeClassifier`` (gini/entropy impurity,
class-frequency leaves) and ``DecisionTreeRegressor`` (variance impurity,
mean-value leaves). They share the split search, the ``_Node`` structure,
and the ``max_features`` / ``sample_weight`` machinery.
"""

from scratchgrad.tree.decision_tree import DecisionTreeClassifier
from scratchgrad.tree.decision_tree_regressor import DecisionTreeRegressor

__all__ = ["DecisionTreeClassifier", "DecisionTreeRegressor"]
