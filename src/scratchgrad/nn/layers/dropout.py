r"""Dropout: randomly zero units during training, scaling to preserve the mean.

Full derivation: docs/derivations/dropout_batchnorm.md section 2.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray
from scratchgrad.utils.validation import check_random_state


def _dropout_mask(
    shape: tuple[int, ...], p: float, rng: np.random.Generator
) -> FloatArray:
    """Draw an inverted-dropout keep mask: ``bernoulli(1-p) / (1-p)``."""
    keep_prob = 1.0 - p
    keep = rng.random(shape) < keep_prob  # True with probability 1-p
    return keep / keep_prob


def _dropout_forward(x: FloatArray, mask: FloatArray) -> FloatArray:
    """Compute ``x * mask``."""
    return x * mask


def _dropout_backward(mask: FloatArray, grad_output: FloatArray) -> FloatArray:
    """Compute ``dL/dx = grad_output * mask`` (same mask as forward)."""
    return grad_output * mask


class Dropout(Module):
    r"""Inverted dropout: zero each unit independently with probability ``p``.

    Training: draws ``mask = bernoulli(1-p) / (1-p)`` and computes
    ``y = x * mask`` -- the ``1/(1-p)`` scaling means ``E[y] = x``, so
    :meth:`eval` needs no compensating rescale (matches
    ``torch.nn.Dropout``). Evaluation: identity, no RNG draw.

    Parameters
    ----------
    p : float, default=0.5
        Probability of zeroing a unit. Must be in ``[0, 1)`` -- ``p=1``
        would divide by zero in the ``1/(1-p)`` scale.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds the mask generator. Unlike ``Linear`` (draws once, at
        construction), this generator is drawn from on every
        training-mode :meth:`forward` call, so its state keeps advancing.

    Raises
    ------
    ValueError
        If ``p`` is not in ``[0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Dropout(p=0.5, random_state=0)
    >>> x = np.ones((3, 4))
    >>> y = layer.forward(x)
    >>> y.shape
    (3, 4)
    >>> _ = layer.eval()
    >>> np.testing.assert_array_equal(layer.forward(x), x)

    """

    def __init__(self, p: float = 0.5, random_state: int | None = None) -> None:
        """See the class docstring for parameter descriptions."""
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self.random_state = random_state
        self._rng = check_random_state(random_state)
        self._mask: FloatArray | None = None

    def forward(self, x: FloatArray) -> FloatArray:
        """Apply a fresh dropout mask in training mode, else the identity."""
        if not self.training:
            self._mask = None
            return x
        self._mask = _dropout_mask(x.shape, self.p, self._rng)
        return _dropout_forward(x, self._mask)

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dx``, reusing the mask cached by :meth:`forward`."""
        if self._mask is None:
            return grad_output
        return _dropout_backward(self._mask, grad_output)
