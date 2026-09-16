r"""Losses: MSE, BCE-with-logits, cross-entropy-with-logits.

Each takes raw scores (``y_pred``, or ``z``/``Z`` logits) directly --
fused with the link function internally for numerical stability, the same
move as ``LogisticRegression``'s ``softplus(z) - yz`` loss. Not a
``Module`` subclass: a loss is always the last op ("root") of a backward
pass, ``forward`` takes predictions *and* targets rather than a single
input, and ``backward`` takes no upstream gradient -- there is nothing
further downstream to receive one. Full derivation:
docs/derivations/nn.md section 4.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray
from scratchgrad.utils.math import logsumexp, sigmoid, softmax


def _mse_forward(y_pred: FloatArray, y_true: FloatArray) -> float:
    """Compute ``mean((y_pred - y_true) ** 2)`` over every entry."""
    return float(np.mean((y_pred - y_true) ** 2))


def _mse_backward(y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
    """Compute ``dL/dy_pred = 2 * (y_pred - y_true) / y_pred.size``."""
    diff = y_pred - y_true
    return 2.0 * diff / diff.size


def _bce_with_logits_forward(z: FloatArray, y: FloatArray) -> float:
    """Compute ``mean(softplus(z) - y * z)``, ``softplus`` via ``logaddexp``."""
    softplus = np.logaddexp(0.0, z)  # softplus(z) = log(1 + e^z)
    return float(np.mean(softplus - y * z))


def _bce_with_logits_backward(z: FloatArray, y: FloatArray) -> FloatArray:
    """Compute ``dL/dz = (sigma(z) - y) / n``."""
    return (sigmoid(z) - y) / z.size


def _cross_entropy_forward(Z: FloatArray, y_true: FloatArray) -> float:
    """Compute ``mean(-log_softmax(Z)[y_true])`` via ``logsumexp``."""
    log_probs = Z - logsumexp(Z, axis=-1)[:, None]  # log softmax(Z), per row
    return float(-np.sum(y_true * log_probs) / Z.shape[0])


def _cross_entropy_backward(Z: FloatArray, y_true: FloatArray) -> FloatArray:
    """Compute ``dL/dZ = (softmax(Z) - y_true) / n``."""
    return (softmax(Z, axis=-1) - y_true) / Z.shape[0]


class MSELoss:
    r"""Mean squared error, :math:`L = \mathrm{mean}((\hat y - y)^2)`.

    The mean is over *every* entry of ``y_pred`` (samples and output
    dimensions alike), matching PyTorch's ``nn.MSELoss()`` default -- the
    ``-m reference`` parity target for this unit.

    Examples
    --------
    >>> import numpy as np
    >>> loss = MSELoss()
    >>> round(loss.forward(np.array([1.0, 2.0]), np.array([0.0, 2.0])), 4)
    0.5

    """

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._y_pred: FloatArray | None = None
        self._y_true: FloatArray | None = None

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> float:
        """Compute the loss, caching inputs for :meth:`backward`."""
        self._y_pred, self._y_true = y_pred, y_true
        return _mse_forward(y_pred, y_true)

    def backward(self) -> FloatArray:
        """Compute ``dL/dy_pred``."""
        return _mse_backward(self._y_pred, self._y_true)


class BCEWithLogitsLoss:
    r"""Binary cross-entropy from logits.

    :math:`L = \mathrm{mean}(\mathrm{softplus}(z) - yz)` -- the same
    stability trick as ``LogisticRegression``'s loss, avoiding
    ``log(sigmoid(z))`` underflow for very negative ``z``.
    """

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._z: FloatArray | None = None
        self._y: FloatArray | None = None

    def forward(self, z: FloatArray, y: FloatArray) -> float:
        """Compute the loss from logits ``z``, caching inputs."""
        self._z, self._y = z, y
        return _bce_with_logits_forward(z, y)

    def backward(self) -> FloatArray:
        """Compute ``dL/dz``."""
        return _bce_with_logits_backward(self._z, self._y)


class CrossEntropyLoss:
    r"""Multiclass cross-entropy from logits.

    :math:`L = \mathrm{mean}_i(-\log \mathrm{softmax}(Z_i)[y_i])`,
    computed via ``logsumexp`` for stability. ``y_true`` is one-hot,
    shape ``(n, n_classes)``.
    """

    def __init__(self) -> None:
        """Initialize with no cached state (set on the first ``forward``)."""
        self._Z: FloatArray | None = None
        self._y_true: FloatArray | None = None

    def forward(self, Z: FloatArray, y_true: FloatArray) -> float:
        """Compute the loss from logits ``Z``, caching inputs."""
        self._Z, self._y_true = Z, y_true
        return _cross_entropy_forward(Z, y_true)

    def backward(self) -> FloatArray:
        """Compute ``dL/dZ``."""
        return _cross_entropy_backward(self._Z, self._y_true)
