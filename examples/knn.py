"""k-NN on interleaving moons: a non-linear boundary, and the role of k.

Logistic regression can only draw a straight boundary, so it tops out around
85% on the two-moons problem. k-NN has no such limit — it follows the local
shape of the data. The k sweep shows the bias-variance tradeoff: small k
overfits the noise, large k oversmooths. Distance weighting lets far
neighbours matter less.

Run:
    uv run python examples/knn.py
"""

from __future__ import annotations

from scratchgrad.datasets import make_moons
from scratchgrad.linear import LogisticRegression
from scratchgrad.neighbors import KNeighborsClassifier
from scratchgrad.preprocessing import StandardScaler, train_test_split


def main() -> None:
    """Fit k-NN on noisy moons, sweep ``k``, compare to a linear baseline."""
    X, y = make_moons(n_samples=600, noise=0.25, random_state=0)
    X = StandardScaler().fit_transform(X)  # k-NN is not scale-invariant
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    baseline = LogisticRegression().fit(X_train, y_train)
    print(
        f"logistic regression (linear boundary): test acc = "
        f"{baseline.score(X_test, y_test):.4f}"
    )

    print("k sweep (uniform weights)")
    for k in (1, 3, 5, 11, 25, 75, 200):
        model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  k={k:<4} train acc={train_acc:.4f}  test acc={test_acc:.4f}")

    print("uniform vs distance weighting (k=25)")
    for weights in ("uniform", "distance"):
        model = KNeighborsClassifier(n_neighbors=25, weights=weights).fit(
            X_train, y_train
        )
        print(f"  {weights:<9} test acc = {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
