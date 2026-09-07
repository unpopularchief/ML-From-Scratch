"""scikit-learn parity for GaussianNB.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs
from scratchgrad.naive_bayes import GaussianNB

pytestmark = pytest.mark.reference


def test_fitted_parameters_match_sklearn() -> None:
    naive_bayes = pytest.importorskip("sklearn.naive_bayes")

    X, y = make_blobs(n_samples=200, n_features=4, centers=3, random_state=0)

    ours = GaussianNB().fit(X, y)
    theirs = naive_bayes.GaussianNB().fit(X, y)

    np.testing.assert_allclose(ours.mean_, theirs.theta_, rtol=1e-6)
    np.testing.assert_allclose(ours.var_, theirs.var_, rtol=1e-6)
    np.testing.assert_allclose(ours.class_prior_, theirs.class_prior_, rtol=1e-6)
    assert ours.epsilon_ == pytest.approx(theirs.epsilon_)


@pytest.mark.parametrize("centers", [2, 3, 5])
def test_predict_and_proba_match_sklearn(centers) -> None:
    naive_bayes = pytest.importorskip("sklearn.naive_bayes")

    X, y = make_blobs(
        n_samples=240,
        n_features=3,
        centers=centers,
        cluster_std=3.0,
        random_state=centers,
    )
    X_query, _ = make_blobs(
        n_samples=60, n_features=3, centers=centers, cluster_std=3.0, random_state=99
    )

    ours = GaussianNB().fit(X, y)
    theirs = naive_bayes.GaussianNB().fit(X, y)

    np.testing.assert_array_equal(ours.predict(X_query), theirs.predict(X_query))
    np.testing.assert_allclose(
        ours.predict_proba(X_query), theirs.predict_proba(X_query), rtol=1e-6
    )
    np.testing.assert_allclose(
        ours.predict_log_proba(X_query), theirs.predict_log_proba(X_query), rtol=1e-6
    )


def test_explicit_priors_match_sklearn() -> None:
    naive_bayes = pytest.importorskip("sklearn.naive_bayes")

    X, y = make_blobs(n_samples=150, n_features=2, centers=3, random_state=1)
    priors = [0.2, 0.3, 0.5]

    ours = GaussianNB(priors=priors).fit(X, y)
    theirs = naive_bayes.GaussianNB(priors=priors).fit(X, y)

    np.testing.assert_allclose(ours.class_prior_, theirs.class_prior_)
    np.testing.assert_array_equal(ours.predict(X), theirs.predict(X))
