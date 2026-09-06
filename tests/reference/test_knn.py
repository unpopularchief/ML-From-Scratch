"""scikit-learn parity for KNeighborsClassifier.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs
from scratchgrad.neighbors import KNeighborsClassifier

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_matches_sklearn(metric, weights) -> None:
    neighbors = pytest.importorskip("sklearn.neighbors")

    X, y = make_blobs(n_samples=200, n_features=4, centers=3, random_state=0)
    X_query, _ = make_blobs(n_samples=50, n_features=4, centers=3, random_state=1)

    ours = KNeighborsClassifier(n_neighbors=7, metric=metric, weights=weights).fit(X, y)
    theirs = neighbors.KNeighborsClassifier(
        n_neighbors=7, metric=metric, weights=weights, algorithm="brute"
    ).fit(X, y)

    np.testing.assert_array_equal(ours.predict(X_query), theirs.predict(X_query))
    np.testing.assert_allclose(
        ours.predict_proba(X_query), theirs.predict_proba(X_query), rtol=1e-6
    )


def test_matches_sklearn_across_k() -> None:
    neighbors = pytest.importorskip("sklearn.neighbors")

    X, y = make_blobs(n_samples=150, n_features=3, centers=4, random_state=2)
    X_query, _ = make_blobs(n_samples=40, n_features=3, centers=4, random_state=3)

    for k in (1, 3, 15, 50):
        ours = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        theirs = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm="brute").fit(
            X, y
        )
        np.testing.assert_array_equal(ours.predict(X_query), theirs.predict(X_query))
