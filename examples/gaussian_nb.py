"""Gaussian naive Bayes: a one-pass generative classifier, and its blind spot.

Part 1 fits GaussianNB on well-separated Gaussian blobs — the model's
assumptions hold, so a single closed-form pass gets near-perfect accuracy.

Part 2 feeds it two features that are near-duplicates of the same latent
signal. The "naive" conditional-independence assumption treats them as two
independent votes and double-counts the evidence: accuracy stays reasonable
but ``predict_proba`` becomes badly overconfident.

Run:
    uv run python examples/gaussian_nb.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import make_blobs
from scratchgrad.naive_bayes import GaussianNB
from scratchgrad.preprocessing import train_test_split
from scratchgrad.utils.validation import check_random_state


def separable_blobs() -> None:
    """Fit on well-separated blobs, where the Gaussian assumption holds."""
    X, y = make_blobs(n_samples=600, centers=3, cluster_std=1.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    model = GaussianNB().fit(X_train, y_train)

    print("well-separated 3-class blobs")
    print(f"  test accuracy = {model.score(X_test, y_test):.4f}")
    print(f"  class priors  = {np.round(model.class_prior_, 3)}")
    for c, mean in zip(model.classes_, model.mean_, strict=True):
        print(f"  class {c:.0f} feature means = {np.round(mean, 2)}")


def correlated_features() -> None:
    """Show predict_proba going overconfident when features are redundant."""
    rng = check_random_state(0)
    z = rng.standard_normal(800)  # the latent signal
    y = (z > 0.0).astype(np.float64)
    x1 = z + 0.7 * rng.standard_normal(800)
    x2 = z + 0.7 * rng.standard_normal(800)  # a near-duplicate of x1
    X = np.column_stack([x1, x2])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    model = GaussianNB().fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    confidence = model.predict_proba(X_test).max(axis=1).mean()
    print("two correlated features (naive independence assumption violated)")
    print(f"  test accuracy        = {accuracy:.4f}")
    print(f"  mean predicted proba = {confidence:.4f}  <- should track accuracy")
    print(f"  overconfidence gap   = {confidence - accuracy:+.4f}")


def main() -> None:
    """Run both demonstrations."""
    separable_blobs()
    print()
    correlated_features()


if __name__ == "__main__":
    main()
