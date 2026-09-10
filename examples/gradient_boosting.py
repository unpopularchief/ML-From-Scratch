"""Gradient boosting on two moons: shallow trees, stacked by the gradient.

A single decision stump barely beats chance on interleaved moons. Gradient
boosting keeps a raw log-odds score, and each round fits a small regression
tree to the negative gradient of the log loss (the residual y - p), refines
its leaf values with one Newton step, and adds `learning_rate` times that to
the score. This script:

1. compares a single stump's test accuracy to the boosted ensemble's;
2. prints the staged test-accuracy curve and the training-deviance curve
   (which keeps dropping even as test accuracy plateaus — boosting can
   overfit, so `n_estimators` is a real hyperparameter);
3. sweeps `learning_rate` to show the shrinkage / n_estimators trade-off;
4. sweeps `subsample` (stochastic gradient boosting).

Run:
    uv run python examples/gradient_boosting.py
"""

from __future__ import annotations

from scratchgrad.datasets import make_moons
from scratchgrad.ensemble import GradientBoostingClassifier
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier


def main() -> None:
    """Boost shallow trees on noisy moons and trace the boosting curves."""
    X, y = make_moons(n_samples=1200, noise=0.25, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    tree = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
    print(f"single decision stump:  test acc={tree.score(X_test, y_test):.4f}")

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(X_train, y_train)
    print(
        f"gradient boosting:      test acc={model.score(X_test, y_test):.4f}  "
        f"(200 trees, lr=0.1)"
    )

    print("\nstaged test accuracy and training deviance")
    staged = [(p == y_test).mean() for p in model.staged_predict(X_test)]
    for t in (1, 5, 10, 25, 50, 100, 150, 200):
        print(
            f"  after {t:>3} rounds  test acc={staged[t - 1]:.4f}  "
            f"train deviance={model.train_score_[t - 1]:.4f}"
        )

    print("\nlearning_rate sweep (n_estimators * learning_rate held ~constant)")
    for lr, n in ((1.0, 30), (0.5, 60), (0.1, 300), (0.05, 600)):
        m = GradientBoostingClassifier(
            n_estimators=n, learning_rate=lr, max_depth=3, random_state=0
        ).fit(X_train, y_train)
        print(
            f"  lr={lr:<5} n={n:<4} train acc={m.score(X_train, y_train):.4f}  "
            f"test acc={m.score(X_test, y_test):.4f}"
        )

    print("\nsubsample sweep (stochastic gradient boosting, 150 rounds)")
    for subsample in (1.0, 0.8, 0.5, 0.3):
        m = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            subsample=subsample,
            random_state=0,
        ).fit(X_train, y_train)
        print(f"  subsample={subsample:<4} test acc={m.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
