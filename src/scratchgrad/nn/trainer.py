"""Trainer: a minimal fit loop -- minibatching, epochs, and a history dict.

No new differentiable math here (unlike every other file in ``nn/``) --
this is the orchestration `plan.md`'s file tree calls out as
``trainer.py # minimal fit loop: batching, epochs, history``, sitting on
top of layers/losses/optimizers that already exist. It exists so
``examples/mnist.py`` (and any future multi-epoch, mini-batched example)
doesn't hand-write its own epoch loop the way ``examples/nn.py``'s small
full-batch loop still does.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray
from scratchgrad.utils.validation import check_random_state

Metric = Callable[[FloatArray, FloatArray], float]


class Trainer:
    r"""Minibatch training loop over a hand-wired chain of ``Module``\ s.

    Owns nothing about the network's *shape* -- ``layers`` is just the
    same flat, ordered ``list[Module]`` ``examples/nn.py`` already
    hand-chains ``forward``/``backward`` over. What ``Trainer`` adds is
    the mechanical part every multi-epoch example would otherwise repeat:
    minibatching, switching every layer to :meth:`Module.train`/
    :meth:`Module.eval` at the right points, and collecting a history.

    Parameters
    ----------
    layers : list of Module
        Applied via ``forward`` in order, ``backward`` in reverse -- the
        same convention ``examples/nn.py`` already uses.
    loss_fn : object
        Anything with ``forward(y_pred, y_true) -> float`` and
        ``backward() -> ndarray`` (e.g. ``BCEWithLogitsLoss``,
        ``CrossEntropyLoss``).
    optimizer : object
        Anything with ``step(params: list[ndarray], grads: list[ndarray])
        -> None`` (any of ``scratchgrad.optim``).

    """

    def __init__(
        self, layers: list[Module], loss_fn: object, optimizer: object
    ) -> None:
        """See the class docstring for parameter descriptions."""
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def _forward(self, x: FloatArray) -> FloatArray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def _backward(self, grad: FloatArray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _set_mode(self, training: bool) -> None:
        for layer in self.layers:
            layer.training = training

    def _step(
        self, x_batch: FloatArray, y_batch: FloatArray
    ) -> tuple[float, FloatArray]:
        """Run one training-mode forward/backward/optimizer step.

        Returns ``(loss, y_pred)`` -- ``y_pred`` is the same training-mode
        prediction the step's gradient came from, reused for
        ``train_metric`` rather than recomputed, so a metric never triggers
        a second ``Dropout`` draw or a second ``BatchNorm1d`` running-stats
        update on the same batch.
        """
        y_pred = self._forward(x_batch)
        loss = self.loss_fn.forward(y_pred, y_batch)
        self._backward(self.loss_fn.backward())
        params = [p for layer in self.layers for p in layer.parameters()]
        grads = [g for layer in self.layers for g in layer.grads()]
        self.optimizer.step(params, grads)
        return loss, y_pred

    def _evaluate(
        self, X: FloatArray, y: FloatArray, metric: Metric | None
    ) -> tuple[float, float | None]:
        """Run one eval-mode forward pass; return ``(loss, metric_value)``."""
        self._set_mode(False)
        y_pred = self._forward(X)
        loss = self.loss_fn.forward(y_pred, y)
        metric_value = metric(y_pred, y) if metric is not None else None
        self._set_mode(True)
        return loss, metric_value

    def fit(
        self,
        X: FloatArray,
        y: FloatArray,
        epochs: int,
        batch_size: int,
        X_val: FloatArray | None = None,
        y_val: FloatArray | None = None,
        metric: Metric | None = None,
        shuffle: bool = True,
        random_state: int | np.random.Generator | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        r"""Train for ``epochs`` passes over ``X``/``y``, in ``batch_size`` minibatches.

        Each epoch: shuffle (if ``shuffle``), run every minibatch through
        one training-mode forward/backward/``optimizer.step``, then --
        only if ``X_val`` is given -- one eval-mode forward pass over the
        validation set. ``train_loss``/``train_metric`` are the
        sample-weighted average over that epoch's *training-mode*
        minibatches (the actual quantity being optimized, at no extra
        forward-pass cost); ``val_loss``/``val_metric`` come from a
        genuine eval-mode pass, since evaluation is exactly what
        :meth:`Module.eval` exists for.

        Parameters
        ----------
        X, y : ndarray
            Training data; ``y`` in whatever shape ``loss_fn`` expects
            (e.g. one-hot for ``CrossEntropyLoss``).
        epochs : int
            Number of passes over the full training set.
        batch_size : int
            Minibatch size; the last batch of an epoch may be smaller.
        X_val, y_val : ndarray or None, default=None
            Held-out data evaluated (in eval mode) once per epoch. Both or
            neither must be given.
        metric : callable or None, default=None
            ``metric(y_pred, y_true) -> float``, e.g. an accuracy computed
            from raw predictions/one-hot targets. Tracked alongside loss
            when given; omitted from ``history`` entirely otherwise.
        shuffle : bool, default=True
            Reshuffle training indices at the start of every epoch.
        random_state : int, numpy.random.Generator, or None, default=None
            Seeds the per-epoch shuffle.
        verbose : bool, default=False
            Print one line per epoch.

        Returns
        -------
        dict[str, list[float]]
            ``"train_loss"`` always; ``"train_metric"`` if ``metric`` is
            given; ``"val_loss"``/``"val_metric"`` if ``X_val`` is given
            (the latter only if ``metric`` is also given).

        Raises
        ------
        ValueError
            If exactly one of ``X_val``/``y_val`` is given.

        """
        if (X_val is None) != (y_val is None):
            raise ValueError("X_val and y_val must both be given, or neither.")

        rng = check_random_state(random_state)
        n = X.shape[0]
        history: dict[str, list[float]] = {"train_loss": []}
        if metric is not None:
            history["train_metric"] = []
        if X_val is not None:
            history["val_loss"] = []
            if metric is not None:
                history["val_metric"] = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(n) if shuffle else np.arange(n)
            total_loss = 0.0
            total_metric = 0.0

            for start in range(0, n, batch_size):
                batch_idx = order[start : start + batch_size]
                x_batch, y_batch = X[batch_idx], y[batch_idx]
                loss, y_pred = self._step(x_batch, y_batch)
                total_loss += loss * len(batch_idx)
                if metric is not None:
                    total_metric += metric(y_pred, y_batch) * len(batch_idx)

            history["train_loss"].append(total_loss / n)
            if metric is not None:
                history["train_metric"].append(total_metric / n)

            log_parts = [
                f"epoch {epoch}/{epochs}",
                f"loss={history['train_loss'][-1]:.4f}",
            ]
            if metric is not None:
                log_parts.append(f"metric={history['train_metric'][-1]:.4f}")

            if X_val is not None:
                val_loss, val_metric_value = self._evaluate(X_val, y_val, metric)
                history["val_loss"].append(val_loss)
                log_parts.append(f"val_loss={val_loss:.4f}")
                if metric is not None:
                    history["val_metric"].append(val_metric_value)
                    log_parts.append(f"val_metric={val_metric_value:.4f}")

            if verbose:
                print(" ".join(log_parts))

        return history
