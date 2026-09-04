"""Ridge regression on collinear data: why the L2 penalty helps.

Two features are made almost identical. Ordinary least squares has nothing
stopping it from putting a large positive weight on one and a large negative
weight on the other, so its coefficients blow up and it generalises poorly.
Ridge's L2 penalty makes that expensive: it splits the weight between the
redundant features and holds the norm down, and its held-out score is better.

Run:
    uv run python examples/ridge.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.linear import LinearRegression, Ridge
from scratchgrad.metrics import r2_score, root_mean_squared_error
from scratchgrad.preprocessing import train_test_split


def main() -> None:
    """Fit OLS and ridge on collinear data, then sweep ``alpha``."""
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.standard_normal(n)
    x2 = x1 + 0.01 * rng.standard_normal(n)  # x2 is nearly a copy of x1
    X = np.column_stack([x1, x2, rng.standard_normal((n, 3))])
    true_w = np.array([1.0, 1.0, 0.5, -0.5, 2.0])
    y = X @ true_w + 0.5 * rng.standard_normal(n)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    ols = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)

    for name, model in (("ordinary least squares", ols), ("ridge (alpha=1.0)", ridge)):
        pred = model.predict(X_test)
        print(name)
        print(f"  coef_       = {np.round(model.coef_, 3)}")
        print(f"  ||coef_||_2 = {np.linalg.norm(model.coef_):.3f}")
        print(f"  test R^2    = {r2_score(y_test, pred):.4f}")
        print(f"  test RMSE   = {root_mean_squared_error(y_test, pred):.4f}")

    print("alpha sweep (test R^2, coefficient norm)")
    for alpha in (0.0, 0.01, 0.1, 1.0, 10.0, 100.0):
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        pred = model.predict(X_test)
        print(
            f"  alpha={alpha:<7} R^2={r2_score(y_test, pred):.4f}  "
            f"||coef_||_2={np.linalg.norm(model.coef_):.3f}"
        )


if __name__ == "__main__":
    main()
