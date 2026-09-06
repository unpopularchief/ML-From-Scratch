"""scikit-learn parity for LogisticRegression.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.linear import LogisticRegression

pytestmark = pytest.mark.reference

# Both sides run to a tight tolerance so the comparison is not dominated by
# either optimiser's stopping rule.
_OURS = {"max_iter": 1000, "tol": 1e-12}
_THEIRS = {"max_iter": 20_000, "tol": 1e-12}


def test_matches_sklearn_with_l2_penalty() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_moons(n_samples=300, noise=0.25, random_state=0)

    ours = LogisticRegression(C=1.0, **_OURS).fit(X, y)
    theirs = linear_model.LogisticRegression(C=1.0, **_THEIRS).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_.ravel(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_[0], rtol=1e-4)
    np.testing.assert_allclose(
        ours.predict_proba(X), theirs.predict_proba(X), rtol=1e-5, atol=1e-6
    )


def test_matches_sklearn_across_C() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_moons(n_samples=250, noise=0.3, random_state=2)

    for C in (0.1, 1.0, 10.0):
        ours = LogisticRegression(C=C, **_OURS).fit(X, y)
        theirs = linear_model.LogisticRegression(C=C, **_THEIRS).fit(X, y)
        np.testing.assert_allclose(
            ours.coef_, theirs.coef_.ravel(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(ours.intercept_, theirs.intercept_[0], rtol=1e-4)


def test_matches_sklearn_without_penalty() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_moons(n_samples=300, noise=0.3, random_state=1)

    ours = LogisticRegression(penalty=None, **_OURS).fit(X, y)
    # sklearn spells "no L2 penalty" as C=inf from v1.8 on (penalty=None is
    # deprecated there); our penalty=None is the same unregularised objective.
    theirs = linear_model.LogisticRegression(C=np.inf, **_THEIRS).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_.ravel(), rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(ours.intercept_, theirs.intercept_[0], rtol=1e-4)


def test_matches_sklearn_without_intercept() -> None:
    linear_model = pytest.importorskip("sklearn.linear_model")

    X, y = make_blobs(n_samples=200, centers=2, cluster_std=3.0, random_state=3)
    X = X - X.mean(axis=0)

    ours = LogisticRegression(fit_intercept=False, **_OURS).fit(X, y)
    theirs = linear_model.LogisticRegression(fit_intercept=False, **_THEIRS).fit(X, y)

    np.testing.assert_allclose(ours.coef_, theirs.coef_.ravel(), rtol=1e-4, atol=1e-5)
