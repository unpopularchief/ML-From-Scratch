"""Random forest on two moons: variance reduction, OOB, and importances.

A single unbounded decision tree overfits — a jagged boundary that changes
a lot when the training data is perturbed. Averaging many trees, each on
its own bootstrap resample and using a random feature subset per split,
keeps the low bias and collapses the variance. This script:

1. sweeps `n_estimators` and prints test accuracy plus the out-of-bag
   estimate at each — OOB tracks the held-out score for free;
2. compares a single tree's test accuracy to the forest's;
3. prints the averaged feature importances next to a single tree's.

Run:
    uv run python examples/random_forest.py
"""

from __future__ import annotations

from scratchgrad.datasets import make_moons
from scratchgrad.ensemble import RandomForestClassifier
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier


def main() -> None:
    """Sweep forest size on noisy moons, then inspect a 100-tree forest."""
    X, y = make_moons(n_samples=800, noise=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    tree = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
    print(f"single unbounded tree: test acc={tree.score(X_test, y_test):.4f}")

    print("\nn_estimators sweep (max_features='sqrt', bootstrap=True)")
    for n_estimators in (1, 5, 10, 25, 50, 100):
        forest = RandomForestClassifier(
            n_estimators=n_estimators, oob_score=True, random_state=0
        ).fit(X_train, y_train)
        print(
            f"  n_estimators={n_estimators:<4} "
            f"test acc={forest.score(X_test, y_test):.4f}  "
            f"oob acc={forest.oob_score_:.4f}"
        )

    forest = RandomForestClassifier(n_estimators=100, random_state=0).fit(
        X_train, y_train
    )
    tree_imp = ", ".join(f"{v:.3f}" for v in tree.feature_importances_)
    forest_imp = ", ".join(f"{v:.3f}" for v in forest.feature_importances_)
    print(f"\nfeature_importances_  tree  = [{tree_imp}]")
    print(f"feature_importances_  forest = [{forest_imp}]")


if __name__ == "__main__":
    main()
