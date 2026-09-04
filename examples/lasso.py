"""Lasso on a sparse signal: the L1 penalty as variable selection.

Twenty features are generated but only three actually drive the target.
Ordinary least squares spreads weight across all twenty (every irrelevant
coefficient is some small non-zero number); Lasso's L1 penalty sets most of
them to *exactly* zero and recovers the three that matter. The alpha sweep
shows the support shrinking as the penalty grows.

Run:
    uv run python examples/lasso.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.linear import Lasso, LinearRegression
from scratchgrad.metrics import r2_score, root_mean_squared_error
from scratchgrad.preprocessing import train_test_split


def main() -> None:
    """Fit OLS and Lasso on a sparse problem, then sweep ``alpha``."""
    rng = np.random.default_rng(0)
    n, d = 300, 20
    X = rng.standard_normal((n, d))
    true_w = np.zeros(d)
    true_w[[2, 7, 13]] = [3.0, -2.0, 1.5]  # only 3 of 20 features matter
    y = X @ true_w + 0.5 * rng.standard_normal(n)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    ols = LinearRegression().fit(X_train, y_train)
    lasso = Lasso(alpha=0.1).fit(X_train, y_train)

    for name, model in (("ordinary least squares", ols), ("lasso (alpha=0.1)", lasso)):
        pred = model.predict(X_test)
        print(name)
        print(f"  non-zero coefs = {int(np.sum(model.coef_ != 0.0))} / {d}")
        print(f"  test R^2       = {r2_score(y_test, pred):.4f}")
        print(f"  test RMSE      = {root_mean_squared_error(y_test, pred):.4f}")

    print(f"true non-zero features: {np.flatnonzero(true_w).tolist()}")
    print(f"lasso selected:         {np.flatnonzero(lasso.coef_).tolist()}")

    print("alpha sweep (non-zeros, test R^2)")
    for alpha in (0.0, 0.01, 0.05, 0.1, 0.5, 1.0):
        model = Lasso(alpha=alpha, max_iter=50_000).fit(X_train, y_train)
        pred = model.predict(X_test)
        print(
            f"  alpha={alpha:<6} nnz={int(np.sum(model.coef_ != 0.0)):>2}  "
            f"R^2={r2_score(y_test, pred):.4f}"
        )


if __name__ == "__main__":
    main()
