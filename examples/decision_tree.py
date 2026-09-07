"""Decision tree on two moons: the overfitting curve and what the tree learned.

A tree with no depth limit memorises the training set (100% train accuracy,
a jagged boundary). Capping `max_depth` trades training fit for
generalisation — the classic bias-variance curve. This script sweeps
`max_depth`, prints train vs test accuracy at each, then shows the fitted
tree as indented text and its feature importances.

Run:
    uv run python examples/decision_tree.py
"""

from __future__ import annotations

from scratchgrad.datasets import make_moons
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier
from scratchgrad.tree.decision_tree import _Node


def print_tree(node: _Node, feature_names: list[str], indent: str = "") -> None:
    """Recursively print a fitted tree as indented ``feature <= threshold`` tests."""
    if node.is_leaf:
        distribution = ", ".join(f"{v:.2f}" for v in node.value)
        print(f"{indent}leaf  n={node.n_samples}  p=[{distribution}]")
        return
    name = feature_names[node.feature]
    print(f"{indent}{name} <= {node.threshold:.3f}  (impurity {node.impurity:.3f})")
    print_tree(node.left, feature_names, indent + "  ")
    print_tree(node.right, feature_names, indent + "  ")


def main() -> None:
    """Sweep tree depth on noisy moons, then inspect a depth-4 tree."""
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    print("max_depth sweep (criterion=gini)")
    for max_depth in (1, 2, 3, 5, 8, None):
        model = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        label = "None" if max_depth is None else str(max_depth)
        print(
            f"  max_depth={label:<4} depth={model.max_depth_:<2} "
            f"train acc={train_acc:.4f}  test acc={test_acc:.4f}"
        )

    print("\nfitted tree (max_depth=4)")
    model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
    print_tree(model.tree_, ["x0", "x1"])
    importances = ", ".join(f"{v:.3f}" for v in model.feature_importances_)
    print(f"\nfeature_importances_ = [{importances}]")


if __name__ == "__main__":
    main()
