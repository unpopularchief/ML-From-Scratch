"""AdaBoost (SAMME) on two moons: how a stack of stumps beats one stump.

A single decision stump is a weak classifier — one axis-aligned cut, well
under 80% on noisy moons. AdaBoost fits a sequence of stumps, each on the
training set reweighted toward the previous stump's mistakes, and votes
them with weights alpha_m = log((1 - err_m) / err_m) + log(K - 1). This
script:

1. compares a single stump's test accuracy to the boosted ensemble's;
2. prints the staged test-accuracy curve (accuracy after each round);
3. shows how the per-round error and alpha evolve as boosting proceeds.

Run:
    uv run python examples/adaboost.py
"""

from __future__ import annotations

from scratchgrad.datasets import make_moons
from scratchgrad.ensemble import AdaBoostClassifier
from scratchgrad.preprocessing import train_test_split
from scratchgrad.tree import DecisionTreeClassifier


def main() -> None:
    """Boost stumps on noisy moons and trace the boosting curve."""
    X, y = make_moons(n_samples=800, noise=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    stump = DecisionTreeClassifier(max_depth=1).fit(X_train, y_train)
    print(f"single stump:      test acc={stump.score(X_test, y_test):.4f}")

    model = AdaBoostClassifier(n_estimators=200).fit(X_train, y_train)
    print(
        f"AdaBoost (200):    test acc={model.score(X_test, y_test):.4f}  "
        f"({len(model.estimators_)} stumps kept)"
    )

    print("\nstaged test accuracy")
    staged = list(model.staged_score(X_test, y_test))
    for t in (1, 2, 5, 10, 25, 50, 100, 200):
        if t <= len(staged):
            print(f"  after {t:>3} rounds  test acc={staged[t - 1]:.4f}")

    print("\nper-round weighted error and alpha")
    for t in (1, 2, 5, 10, 25, 50):
        if t <= len(model.estimators_):
            err = model.estimator_errors_[t - 1]
            alpha = model.estimator_weights_[t - 1]
            print(f"  round {t:>3}  err={err:.4f}  alpha={alpha:.4f}")


if __name__ == "__main__":
    main()
