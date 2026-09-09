"""scikit-learn parity for AdaBoostClassifier (SAMME).

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

SAMME on a depth-1 ``DecisionTreeClassifier`` uses no randomness, and our
stump's deterministic tie-break matches scikit-learn's on continuous data
where the best split is unique. So the errors, the estimator weights, the
predictions, and ``predict_proba`` all match to a tight tolerance — much
closer than the random-forest parity, which only compares held-out
accuracy. A small tolerance is still allowed for the rare equal-gain stump
split that our lowest-index rule and scikit-learn's RNG break differently.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.ensemble import AdaBoostClassifier

pytestmark = pytest.mark.reference


def _sklearn_adaboost(n_estimators, learning_rate):
    ensemble = pytest.importorskip("sklearn.ensemble")
    tree = pytest.importorskip("sklearn.tree")
    return ensemble.AdaBoostClassifier(
        estimator=tree.DecisionTreeClassifier(max_depth=1),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=0,
    )


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("binary", make_moons(n_samples=400, noise=0.3, random_state=0)),
        (
            "multiclass",
            make_blobs(
                n_samples=360, n_features=4, centers=3, cluster_std=3.0, random_state=0
            ),
        ),
    ],
)
@pytest.mark.parametrize("learning_rate", [1.0, 0.5])
def test_matches_sklearn_samme(name, data, learning_rate) -> None:
    X, y = data
    ours = AdaBoostClassifier(n_estimators=25, learning_rate=learning_rate).fit(X, y)
    theirs = _sklearn_adaboost(25, learning_rate).fit(X, y)

    rounds = min(len(ours.estimators_), len(theirs.estimators_))
    np.testing.assert_allclose(
        ours.estimator_errors_[:rounds], theirs.estimator_errors_[:rounds], atol=1e-9
    )
    np.testing.assert_allclose(
        ours.estimator_weights_[:rounds], theirs.estimator_weights_[:rounds], atol=1e-9
    )

    agree = np.mean(ours.predict(X) == theirs.predict(X))
    assert agree >= 0.97
    assert np.max(np.abs(ours.predict_proba(X) - theirs.predict_proba(X))) < 0.05


def test_decision_function_sign_matches_sklearn_binary() -> None:
    X, y = make_moons(n_samples=300, noise=0.3, random_state=1)
    ours = AdaBoostClassifier(n_estimators=20).fit(X, y)
    theirs = _sklearn_adaboost(20, 1.0).fit(X, y)
    assert (
        np.mean(
            np.sign(ours.decision_function(X)) == np.sign(theirs.decision_function(X))
        )
        >= 0.97
    )
