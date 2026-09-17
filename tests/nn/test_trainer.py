"""Tests for scratchgrad.nn.Trainer.

Tiers (plan.md section 3): the X_val/y_val contract, the history dict's
shape depending on which optional arguments are given, a regression test
for the "reuse the training-mode y_pred, don't recompute" invariant
(BatchNorm1d's running stats must update exactly once per batch), and an
end-to-end convergence check.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.datasets import make_moons
from scratchgrad.metrics import accuracy_score
from scratchgrad.nn import (
    BatchNorm1d,
    BCEWithLogitsLoss,
    Dropout,
    Linear,
    MSELoss,
    ReLU,
    Trainer,
)
from scratchgrad.optim import SGD, Adam


class TestContract:
    def test_only_x_val_raises(self, rng: np.random.Generator) -> None:
        trainer = Trainer([Linear(2, 1, random_state=0)], MSELoss(), SGD(lr=0.01))
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 1))
        with pytest.raises(ValueError, match="X_val"):
            trainer.fit(X, y, epochs=1, batch_size=5, X_val=X)

    def test_only_y_val_raises(self, rng: np.random.Generator) -> None:
        trainer = Trainer([Linear(2, 1, random_state=0)], MSELoss(), SGD(lr=0.01))
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 1))
        with pytest.raises(ValueError, match="y_val"):
            trainer.fit(X, y, epochs=1, batch_size=5, y_val=y)

    def test_history_has_only_train_loss_by_default(
        self, rng: np.random.Generator
    ) -> None:
        trainer = Trainer([Linear(2, 1, random_state=0)], MSELoss(), SGD(lr=0.01))
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 1))
        history = trainer.fit(X, y, epochs=3, batch_size=5)
        assert set(history) == {"train_loss"}
        assert len(history["train_loss"]) == 3

    def test_history_includes_metric_keys_when_metric_given(
        self, rng: np.random.Generator
    ) -> None:
        trainer = Trainer([Linear(2, 1, random_state=0)], MSELoss(), SGD(lr=0.01))
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 1))
        history = trainer.fit(
            X, y, epochs=2, batch_size=5, metric=lambda y_pred, y_true: 0.0
        )
        assert set(history) == {"train_loss", "train_metric"}

    def test_history_includes_val_keys_when_x_val_given(
        self, rng: np.random.Generator
    ) -> None:
        trainer = Trainer([Linear(2, 1, random_state=0)], MSELoss(), SGD(lr=0.01))
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 1))
        X_val = rng.standard_normal((4, 2))
        y_val = rng.standard_normal((4, 1))
        history = trainer.fit(
            X,
            y,
            epochs=2,
            batch_size=5,
            X_val=X_val,
            y_val=y_val,
            metric=lambda y_pred, y_true: 0.0,
        )
        assert set(history) == {"train_loss", "train_metric", "val_loss", "val_metric"}

    def test_layers_end_in_training_mode(self, rng: np.random.Generator) -> None:
        linear = Linear(2, 1, random_state=0)
        trainer = Trainer([linear], MSELoss(), SGD(lr=0.01))
        X_val = rng.standard_normal((4, 2))
        y_val = rng.standard_normal((4, 1))
        trainer.fit(
            rng.standard_normal((10, 2)),
            rng.standard_normal((10, 1)),
            epochs=2,
            batch_size=5,
            X_val=X_val,
            y_val=y_val,
        )
        assert linear.training is True

    def test_verbose_prints_one_line_per_epoch(
        self, rng: np.random.Generator, capsys: pytest.CaptureFixture[str]
    ) -> None:
        trainer = Trainer([Linear(2, 1, random_state=0)], MSELoss(), SGD(lr=0.01))
        X = rng.standard_normal((10, 2))
        y = rng.standard_normal((10, 1))
        trainer.fit(X, y, epochs=3, batch_size=5, verbose=True)
        out_lines = capsys.readouterr().out.strip().splitlines()
        assert len(out_lines) == 3
        assert out_lines[0].startswith("epoch 1/3")


class TestBatchNormRunningStatsUpdateOnce:
    def test_running_mean_matches_a_single_momentum_update(self) -> None:
        # If train_metric recomputed y_pred with a second forward pass
        # instead of reusing the step's own prediction, BatchNorm1d's
        # running stats would be updated twice per batch instead of once.
        bn = BatchNorm1d(3, momentum=0.1)
        trainer = Trainer([bn], MSELoss(), SGD(lr=0.01))
        X = np.array(
            [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [5.0, 6.0, 7.0], [7.0, 8.0, 9.0]]
        )
        y = np.zeros_like(X)

        trainer.fit(
            X, y, epochs=1, batch_size=4, shuffle=False, metric=lambda p, t: 0.0
        )

        expected_running_mean = 0.1 * X.mean(axis=0)  # (1-momentum)*0 + momentum*mu
        np.testing.assert_allclose(bn.running_mean, expected_running_mean)


class TestIntegration:
    def test_training_loss_decreases_on_a_toy_linear_regression(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.standard_normal((200, 3))
        true_w = np.array([[1.5], [-2.0], [0.5]])
        y = X @ true_w

        trainer = Trainer([Linear(3, 1, random_state=0)], MSELoss(), Adam(lr=0.1))
        history = trainer.fit(X, y, epochs=50, batch_size=32, random_state=0)

        assert history["train_loss"][-1] < history["train_loss"][0] * 0.1

    def test_convergence_on_make_moons_with_dropout_and_batchnorm(self) -> None:
        X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
        y = y.astype(np.float64).reshape(-1, 1)

        layers = [
            Linear(2, 16, random_state=0),
            BatchNorm1d(16),
            ReLU(),
            Dropout(p=0.1, random_state=0),
            Linear(16, 1, random_state=1),
        ]
        trainer = Trainer(layers, BCEWithLogitsLoss(), Adam(lr=0.05))

        def metric(y_pred: np.ndarray, y_true: np.ndarray) -> float:
            return accuracy_score(
                y_true.ravel(), (y_pred.ravel() > 0).astype(np.float64)
            )

        history = trainer.fit(
            X, y, epochs=100, batch_size=32, metric=metric, random_state=0
        )

        assert history["train_metric"][-1] > 0.9
