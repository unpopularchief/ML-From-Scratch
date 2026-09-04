"""scikit-learn parity for LinearRegression.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_regression
from scratchgrad.linear import LinearRegression

pytestmark = pytest.mark.reference


def test_matches_sklearn_on_a_noisy_problem() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_regression(n_samples=200, n_features=5, noise=0.3, random_state=0)

    ours = LinearRegression().fit(X, y)
    theirs = linear_model.LinearRegression().fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-6)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, rtol=1e-6)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-6)


def test_matches_sklearn_without_intercept() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_regression(n_samples=150, n_features=4, noise=0.1, random_state=1)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    ours = LinearRegression(fit_intercept=False).fit(X, y)
    theirs = linear_model.LinearRegression(fit_intercept=False).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-6)
