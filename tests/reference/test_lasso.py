"""scikit-learn parity for Lasso.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_regression
from scratchgrad.linear import Lasso

pytestmark = pytest.mark.reference

# Both sides run to a tight tolerance so the comparison is not dominated by
# either optimiser's stopping rule.
_KW = {"max_iter": 200_000, "tol": 1e-9}


def test_matches_sklearn_on_a_noisy_problem() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_regression(n_samples=200, n_features=6, noise=0.3, random_state=0)

    ours = Lasso(alpha=0.5, **_KW).fit(X, y)
    theirs = linear_model.Lasso(alpha=0.5, **_KW).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=1e-4)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_, atol=1e-4)
    np.testing.assert_allclose(ours.predict(X), theirs.predict(X), rtol=1e-4)


def test_matches_sklearn_across_alphas() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_regression(n_samples=150, n_features=8, noise=0.2, random_state=2)

    for alpha in (0.05, 0.2, 1.0, 5.0):
        ours = Lasso(alpha=alpha, **_KW).fit(X, y)
        theirs = linear_model.Lasso(alpha=alpha, **_KW).fit(X, y)
        np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=1e-4)
        np.testing.assert_allclose(ours.intercept_, theirs.intercept_, atol=1e-4)
        # same support (set of selected features)
        np.testing.assert_array_equal(ours.coef_ != 0, theirs.coef_ != 0)


def test_matches_sklearn_without_intercept() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_regression(n_samples=150, n_features=5, noise=0.1, random_state=1)
    X = X - X.mean(axis=0)
    y = y - y.mean()

    ours = Lasso(alpha=0.3, fit_intercept=False, **_KW).fit(X, y)
    theirs = linear_model.Lasso(alpha=0.3, fit_intercept=False, **_KW).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_, atol=1e-4)
