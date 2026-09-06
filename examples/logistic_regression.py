"""Logistic regression on two Gaussian blobs: the sigmoid decision boundary.

Two clusters are generated with enough overlap that they are not linearly
separable. Logistic regression fits a linear log-odds score by maximum
likelihood (Newton / IRLS by default), turns it into a probability with the
sigmoid, and predicts the more likely class. The C sweep shows the L2 penalty
trading coefficient norm (confidence) against fit.

Run:
    uv run python examples/logistic_regression.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import make_blobs
from scratchgrad.linear import LogisticRegression
from scratchgrad.metrics import accuracy_score
from scratchgrad.preprocessing import StandardScaler, train_test_split


def main() -> None:
    """Fit logistic regression on overlapping blobs, then sweep ``C``."""
    X, y = make_blobs(
        n_samples=400, n_features=2, centers=2, cluster_std=4.0, random_state=0
    )
    X = StandardScaler().fit_transform(X)  # so the C sweep and GD are comparable
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    model = LogisticRegression(C=1.0).fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    print("logistic regression (C=1.0, solver='newton')")
    print(f"  coef_       = {np.round(model.coef_, 3)}")
    print(f"  intercept_  = {model.intercept_:.3f}")
    print(f"  n_iter_     = {model.n_iter_}")
    print(f"  train acc   = {accuracy_score(y_train, model.predict(X_train)):.4f}")
    print(f"  test acc    = {accuracy_score(y_test, model.predict(X_test)):.4f}")
    print(f"  mean P(y=1) on true positives = {proba[y_test == 1].mean():.3f}")
    print(f"  mean P(y=1) on true negatives = {proba[y_test == 0].mean():.3f}")

    print("C sweep (test accuracy, coefficient norm)")
    for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
        swept = LogisticRegression(C=C).fit(X_train, y_train)
        acc = accuracy_score(y_test, swept.predict(X_test))
        norm = np.linalg.norm(swept.coef_)
        print(f"  C={C:<8} acc={acc:.4f}  ||coef_||_2={norm:.3f}")

    gd = LogisticRegression(solver="gd", lr=0.1, max_iter=50_000).fit(X_train, y_train)
    print("gradient descent reaches the same solution:")
    print(f"  newton coef_ = {np.round(model.coef_, 3)}")
    print(f"  gd     coef_ = {np.round(gd.coef_, 3)}")


if __name__ == "__main__":
    main()
