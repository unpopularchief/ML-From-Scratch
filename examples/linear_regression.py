"""LinearRegression on synthetic data: the normal equation vs. gradient descent.

Both solvers minimize the same mean-squared-error objective; the normal
equation solves it in one linear-algebra step, gradient descent walks down to
it. On standardized features they land in the same place.

Run:
    uv run python examples/linear_regression.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import make_regression
from scratchgrad.linear import LinearRegression
from scratchgrad.metrics import r2_score, root_mean_squared_error
from scratchgrad.preprocessing import StandardScaler, train_test_split


def main() -> None:
    """Fit both solvers on the same data and print their coefficients and scores."""
    X, y = make_regression(n_samples=200, n_features=3, noise=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0
    )

    # --- Closed form: the normal equation (X^T X) theta = X^T y ---
    ols = LinearRegression(solver="normal").fit(X_train, y_train)
    pred = ols.predict(X_test)
    print("normal equation")
    print(f"  coef_      = {np.round(ols.coef_, 4)}")
    print(f"  intercept_ = {ols.intercept_:.4f}")
    print(f"  test R^2   = {r2_score(y_test, pred):.4f}")
    print(f"  test RMSE  = {root_mean_squared_error(y_test, pred):.4f}")

    # --- Gradient descent: same objective, iterative. Needs scaled features. ---
    scaler = StandardScaler().fit(X_train)
    gd = LinearRegression(solver="gd", lr=0.1, max_iter=10_000, tol=1e-9)
    gd.fit(scaler.transform(X_train), y_train)
    gd_pred = gd.predict(scaler.transform(X_test))
    print("gradient descent (standardized features)")
    print(f"  n_iter_    = {gd.n_iter_}")
    print(f"  test R^2   = {r2_score(y_test, gd_pred):.4f}")
    print(f"  test RMSE  = {root_mean_squared_error(y_test, gd_pred):.4f}")


if __name__ == "__main__":
    main()
