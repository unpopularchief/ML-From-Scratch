"""scikit-learn parity for RandomForestClassifier.

Opt-in (plan.md section 3, tier 4): deselected by the default
``-m 'not reference'`` in pyproject.toml. Run explicitly with::

    uv run pytest -m reference -q

which needs the ``reference`` extra installed (``uv sync --extra reference``).

Our per-node feature draws use a NumPy ``Generator``; scikit-learn's use
its internal C RNG. The two streams diverge, so a matched ``random_state``
does *not* give matched trees or matched predictions — only comparable
held-out accuracy and a comparable out-of-bag score. These are therefore
tolerance checks, not exact ones (contrast ``test_decision_tree.py``,
whose deterministic path *is* exact).
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_blobs, make_moons
from scratchgrad.ensemble import RandomForestClassifier

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_holdout_accuracy_tracks_sklearn(criterion) -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")

    X, y = make_moons(n_samples=600, noise=0.3, random_state=0)
    X_test, y_test = make_moons(n_samples=400, noise=0.3, random_state=1)

    ours = RandomForestClassifier(
        n_estimators=100, criterion=criterion, random_state=0
    ).fit(X, y)
    theirs = ensemble.RandomForestClassifier(
        n_estimators=100, criterion=criterion, random_state=0
    ).fit(X, y)

    ours_acc = np.mean(ours.predict(X_test) == y_test)
    theirs_acc = theirs.score(X_test, y_test)
    assert abs(ours_acc - theirs_acc) < 0.05


def test_multiclass_holdout_accuracy_tracks_sklearn() -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")

    X, y = make_blobs(
        n_samples=500, n_features=6, centers=4, cluster_std=4.0, random_state=0
    )
    X_test, y_test = make_blobs(
        n_samples=300, n_features=6, centers=4, cluster_std=4.0, random_state=1
    )

    ours = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=0)
    ours.fit(X, y)
    theirs = ensemble.RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=0
    ).fit(X, y)

    ours_acc = np.mean(ours.predict(X_test) == y_test)
    assert abs(ours_acc - theirs.score(X_test, y_test)) < 0.05


def test_oob_score_tracks_sklearn() -> None:
    ensemble = pytest.importorskip("sklearn.ensemble")

    X, y = make_blobs(
        n_samples=600, n_features=5, centers=3, cluster_std=4.0, random_state=2
    )

    ours = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0).fit(
        X, y
    )
    theirs = ensemble.RandomForestClassifier(
        n_estimators=200, oob_score=True, random_state=0
    ).fit(X, y)

    assert abs(ours.oob_score_ - theirs.oob_score_) < 0.05
