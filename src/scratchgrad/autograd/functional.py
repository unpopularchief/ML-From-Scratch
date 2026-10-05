r"""Stable softmax, cross-entropy and friends, composed from Tensor ops.

None of these defines a backward: each is a forward composition of
primitives in ``ops.py``, so :meth:`Tensor.backward` differentiates it.
The max-shifts below are constants (no grad path), which is valid because
the shifted expression equals the unshifted one for every shift.

Full derivation: docs/derivations/autograd_layers.md section 2
(``layer_norm`` and ``gelu``: docs/derivations/transformer_block.md sections 1-2).
"""

from __future__ import annotations

import math

import numpy as np

from scratchgrad.autograd.tensor import Tensor


def log_softmax(z: Tensor) -> Tensor:
    r""":math:`\log\operatorname{softmax}(z) = z - \operatorname{lse}(z)`, per row."""
    shift = Tensor(z.data.max(axis=-1, keepdims=True))  # constant, no grad path
    lse = (z - shift).exp().sum(axis=-1, keepdims=True).log() + shift
    return z - lse


def softmax(z: Tensor) -> Tensor:
    """Row-wise softmax, ``exp(log_softmax(z))``."""
    return log_softmax(z).exp()


def cross_entropy(logits: Tensor, y_onehot: Tensor) -> Tensor:
    r"""Mean cross-entropy from logits and one-hot targets.

    :math:`L = -\frac1n \sum_{i,j} Y_{ij} \log\operatorname{softmax}(Z_i)_j`
    (same convention as ``nn.CrossEntropyLoss``).

    Examples
    --------
    >>> z = Tensor([[0.0, 0.0]], requires_grad=True)
    >>> loss = cross_entropy(z, Tensor([[1.0, 0.0]]))
    >>> round(float(loss.data), 4)
    0.6931

    """
    n = logits.shape[0]
    return -(y_onehot * log_softmax(logits)).sum() / n


def bce_with_logits(z: Tensor, y: Tensor) -> Tensor:
    r"""Mean binary cross-entropy from logits, :math:`\mathrm{softplus}(z) - yz`.

    :math:`\mathrm{softplus}(z) = m + \log(e^{-m} + e^{z-m})`, with
    :math:`m = \max(z, 0)` held constant so neither exponent overflows.
    """
    m = Tensor(np.maximum(z.data, 0.0))  # constant, no grad path
    softplus = m + ((-m).exp() + (z - m).exp()).log()
    return (softplus - y * z).mean()


def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Mean squared error over every element."""
    return ((y_pred - y_true) ** 2).mean()


def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    r"""Normalize over the last axis, then scale and shift.

    :math:`y = \gamma \odot (x - \mu)/\sqrt{\sigma^2 + \epsilon} + \beta` with the
    biased per-row variance, as in ``torch.nn.functional.layer_norm``.

    Examples
    --------
    >>> x = Tensor([[1.0, 2.0, 3.0]])
    >>> y = layer_norm(x, Tensor([1.0, 1.0, 1.0]), Tensor([0.0, 0.0, 0.0]))
    >>> round(float(y.data.mean()), 6)
    0.0

    """
    centered = x - x.mean(axis=-1, keepdims=True)
    var = (centered * centered).mean(axis=-1, keepdims=True)
    return centered * (var + eps) ** -0.5 * gamma + beta


def gelu(x: Tensor) -> Tensor:
    r"""Tanh approximation of GELU (the GPT-2 form).

    :math:`\tfrac12 x\,(1 + \tanh(\sqrt{2/\pi}\,(x + 0.044715\,x^3)))`.

    Examples
    --------
    >>> float(gelu(Tensor([0.0])).data[0])
    0.0

    """
    inner = (x + x**3 * 0.044715) * math.sqrt(2.0 / math.pi)
    return x * 0.5 * (inner.tanh() + 1.0)
