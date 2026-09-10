"""scikit-learn parity for GradientBoostingClassifier.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

**No exact parity.** ``sklearn.ensemble.GradientBoostingClassifier`` splits
its base trees on ``criterion="friedman_mse"`` over an RNG-permuted feature
order; our ``DecisionTreeRegressor`` uses ``squared_error`` and the
deterministic lowest-index tie-break. So the trees differ tree-for-tree,
and the comparison is on outcomes: held-out accuracy, held-out log loss,
and ``predict_proba`` in the sup norm, each within a tolerance — the same
level as the ``RandomForest`` reference. In practice the deviance the two
converge toward is close enough that these tolerances are comfortable.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.ensemble import GradientBoostingClassifier
from scratchgrad.preprocessing import train_test_split

pytestmark = pytest.mark.reference


def _log_loss(y_true, proba, classes):
    """Mean negative log-likelihood, without importing sklearn.metrics."""
    idx = np.searchsorted(classes, y_true)
    picked = proba[np.arange(len(y_true)), idx]
    return float(-np.mean(np.log(np.clip(picked, 1e-15, 1.0))))


@pytest.mark.parametrize(
    ("name", "data"),
    [
        ("binary", make_moons(n_samples=500, noise=0.3, random_state=0)),
        (
            "multiclass",
            make_blobs(
                n_samples=480, n_features=4, centers=3, cluster_std=3.0, random_state=0
            ),
        ),
    ],
)
@pytest.mark.parametrize("learning_rate", [0.1, 0.5])
def test_tracks_sklearn(name, data, learning_rate) -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")
    X, y = data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ours = GradientBoostingClassifier(
        n_estimators=100, learning_rate=learning_rate, max_depth=3, random_state=0
    ).fit(X_tr, y_tr)
    theirs = ensemble.GradientBoostingClassifier(
        n_estimators=100, learning_rate=learning_rate, max_depth=3, random_state=0
    ).fit(X_tr, y_tr)

    assert abs(ours.score(X_te, y_te) - theirs.score(X_te, y_te)) < 0.05

    classes = ours.classes_
    our_ll = _log_loss(y_te, ours.predict_proba(X_te), classes)
    their_ll = _log_loss(y_te, theirs.predict_proba(X_te), classes)
    assert our_ll < their_ll + 0.1

    # the two build different trees (friedman_mse vs squared_error), so a
    # handful of points land in a differently-confident leaf; the posterior
    # agrees closely on average even when a few rows diverge
    proba_gap = np.abs(ours.predict_proba(X_te) - theirs.predict_proba(X_te))
    assert proba_gap.mean() < 0.02
    assert proba_gap.max() < 0.3

    assert ours.train_score_.shape == theirs.train_score_.shape


def test_stochastic_gradient_boosting_tracks_sklearn() -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")
    X, y = make_moons(n_samples=800, noise=0.3, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    ours = GradientBoostingClassifier(
        n_estimators=120, subsample=0.5, max_depth=3, random_state=0
    ).fit(X_tr, y_tr)
    theirs = ensemble.GradientBoostingClassifier(
        n_estimators=120, subsample=0.5, max_depth=3, random_state=0
    ).fit(X_tr, y_tr)

    assert abs(ours.score(X_te, y_te) - theirs.score(X_te, y_te)) < 0.06
