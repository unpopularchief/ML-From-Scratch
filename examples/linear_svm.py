"""Linear SVM on two Gaussian blobs: the max-margin decision boundary.

Fits the soft-margin primal objective directly by subgradient descent (no
kernel, no dual). Reports the margin width ``2 / ||w||_2`` and sweeps ``C``
to show the classic margin-vs-violations trade-off: a larger ``C`` penalises
margin violations more, producing a larger ``||w||`` (narrower margin) that
fits the training data more tightly.

Run:
    uv run python examples/linear_svm.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import make_blobs
from scratchgrad.metrics import accuracy_score
from scratchgrad.preprocessing import StandardScaler, train_test_split
from scratchgrad.svm import LinearSVM


def main() -> None:
    """Fit LinearSVM on separable blobs, then sweep ``C``."""
    X, y = make_blobs(
        n_samples=400, n_features=2, centers=2, cluster_std=1.5, random_state=0
    )
    X = StandardScaler().fit_transform(X)  # so the C sweep is comparable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = LinearSVM(C=1.0, max_iter=2000).fit(X_train, y_train)
    margin = 2.0 / np.linalg.norm(model.coef_)

    print("LinearSVM (C=1.0)")
    print(f"  coef_       = {np.round(model.coef_, 3)}")
    print(f"  intercept_  = {model.intercept_:.3f}")
    print(f"  n_iter_     = {model.n_iter_}")
    print(f"  margin (2/||w||) = {margin:.3f}")
    print(f"  train acc   = {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"  test acc    = {accuracy_score(y_test, model.predict(X_test)):.4f}")

    print("C sweep (test accuracy, ||coef_||, margin)")
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        swept = LinearSVM(C=C, max_iter=2000).fit(X_train, y_train)
        acc = accuracy_score(y_test, swept.predict(X_test))
        norm = np.linalg.norm(swept.coef_)
        swept_margin = 2.0 / norm
        print(
            f"  C={C:<8} acc={acc:.4f}  ||coef_||_2={norm:.3f}  "
            f"margin={swept_margin:.3f}"
        )


if __name__ == "__main__":
    main()
