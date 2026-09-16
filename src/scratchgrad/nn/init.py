r"""Weight initialization schemes: zeros, Xavier/Glorot, He/Kaiming.

Full derivation: docs/derivations/nn.md section 5.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray


def zeros(shape: tuple[int, ...]) -> FloatArray:
    """All-zero initialization.

    Used for a layer's bias, which has no symmetry problem. Never used
    for a weight matrix feeding more than one output unit -- every unit
    would then compute an identical function of the input and receive an
    identical gradient forever (docs/derivations/nn.md section 5).

    Parameters
    ----------
    shape : tuple of int
        Output array shape.

    Returns
    -------
    ndarray of shape ``shape``, dtype float64, all zero.

    """
    return np.zeros(shape, dtype=np.float64)


def xavier_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> FloatArray:
    r"""Glorot/Xavier uniform initialization (Glorot & Bengio, 2010).

    :math:`W \sim U(-a, a)`, :math:`a = \sqrt{6 / (\mathrm{fan\_in} +
    \mathrm{fan\_out})}` -- chosen so both the forward-pass activation
    variance and the backward-pass gradient variance are preserved across
    the layer, under a linear/tanh-like activation
    (docs/derivations/nn.md section 5).

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    ndarray of shape ``(fan_in, fan_out)``, dtype float64.

    """
    a = np.sqrt(6.0 / (fan_in + fan_out))  # a = sqrt(6 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out)).astype(np.float64)


def he_normal(fan_in: int, fan_out: int, rng: np.random.Generator) -> FloatArray:
    r"""He/Kaiming normal initialization (He, Zhang, Ren & Sun, 2015).

    :math:`W \sim \mathcal{N}(0, 2/\mathrm{fan\_in})` -- Xavier's
    forward-variance argument adjusted for ReLU, which zeroes out half its
    input in expectation and so needs twice Xavier's variance to preserve
    the forward activation variance (docs/derivations/nn.md section 5).

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.
    rng : numpy.random.Generator
        Source of randomness.

    Returns
    -------
    ndarray of shape ``(fan_in, fan_out)``, dtype float64.

    """
    std = np.sqrt(2.0 / fan_in)  # std = sqrt(2 / fan_in)
    return (rng.standard_normal(size=(fan_in, fan_out)) * std).astype(np.float64)
