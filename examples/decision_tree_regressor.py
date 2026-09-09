"""Regression tree on a noisy sine: the overfitting curve and a step fit.

A regression tree predicts a piecewise-constant function — the mean target
of each leaf's region. With no depth limit it carves one tiny region per
training point (train R^2 = 1, a spiky staircase that generalises badly);
capping `max_depth` trades training fit for smoothness. This script sweeps
`max_depth` on a noisy sine, prints train vs test R^2 at each, shows the
fitted tree as indented text and its feature importances, then fits a clean
1-D step function to show the piecewise-constant shape exactly.

Run:
    uv run python examples/decision_tree_regressor.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeRegressor
from scratchgrad.tree.decision_tree import _Node


def print_tree(node: _Node, feature_names: list[str], indent: str = "") -> None:
    """Recursively print a fitted tree as indented ``feature <= threshold`` tests."""
    if node.is_leaf:
        print(f"{indent}leaf  n={node.n_samples}  predict={node.value:.3f}")
        return
    name = feature_names[node.feature]
    print(f"{indent}{name} <= {node.threshold:.3f}  (variance {node.impurity:.3f})")
    print_tree(node.left, feature_names, indent + "  ")
    print_tree(node.right, feature_names, indent + "  ")


def main() -> None:
    """Sweep tree depth on a noisy sine, then inspect a step-function fit."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-3.0, 3.0, size=(600, 1))
    y = np.sin(2.0 * x[:, 0]) + 0.15 * rng.standard_normal(600)
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0
    )

    print("max_depth sweep (criterion=squared_error)")
    for max_depth in (1, 2, 3, 5, 8, None):
        model = DecisionTreeRegressor(max_depth=max_depth).fit(X_train, y_train)
        label = "None" if max_depth is None else str(max_depth)
        print(
            f"  max_depth={label:<4} depth={model.max_depth_:<2} "
            f"train R2={model.score(X_train, y_train):.4f}  "
            f"test R2={model.score(X_test, y_test):.4f}"
        )

    print("\nfitted tree (max_depth=3)")
    model = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
    print_tree(model.tree_, ["x"])

    # A second feature of pure noise: its importance should stay near zero.
    X2 = np.column_stack([x[:, 0], rng.standard_normal(600)])
    model2 = DecisionTreeRegressor(max_depth=5).fit(X2, y)
    importances = ", ".join(f"{v:.3f}" for v in model2.feature_importances_)
    print(f"\nfeature_importances_ (signal, noise) = [{importances}]")

    # A clean 1-D step function: the tree recovers it exactly, so predict is
    # constant on each true segment.
    print("\nstep function y = [-2, 3, 1] on x < 3, 3 <= x < 7, x >= 7")
    xs = rng.uniform(0.0, 10.0, size=(400, 1))
    ys = np.where(xs[:, 0] < 3.0, -2.0, np.where(xs[:, 0] < 7.0, 3.0, 1.0))
    step = DecisionTreeRegressor().fit(xs, ys)
    for query in (1.0, 5.0, 9.0):
        pred = step.predict(np.array([[query]]))[0]
        print(f"  x={query:<4} predict={pred:+.3f}")


if __name__ == "__main__":
    main()
