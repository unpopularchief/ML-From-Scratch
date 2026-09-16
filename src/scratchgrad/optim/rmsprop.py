r"""RMSprop — per-coordinate learning rate scaled by recent gradient magnitude.

.. math::
    s_{t+1} = \beta s_t + (1-\beta) g_t^2, \qquad
    \theta_{t+1} = \theta_t - \eta\, \frac{g_t}{\sqrt{s_{t+1}} + \epsilon}

Full derivation: ``docs/derivations/optim.md`` section 5.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray


class RMSprop:
    r"""RMSprop (Hinton, 2012 — Coursera lecture 6e).

    :math:`s` is a running average of squared gradients, an elementwise
    estimate of each coordinate's recent gradient scale. Dividing by
    :math:`\sqrt{s_{t+1}}` shrinks the step where gradients are large or
    noisy and relatively grows it where they are small, so a single
    global :math:`\eta` still adapts per coordinate.

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate :math:`\eta`. Must be positive.
    beta : float, default=0.9
        Decay :math:`\beta \in (0, 1)` for the squared-gradient average.
    eps : float, default=1e-8
        Added to :math:`\sqrt{s_{t+1}}` before dividing, so a coordinate
        whose gradient has been exactly ``0`` so far does not divide by
        zero. Must be positive.

    Attributes
    ----------
    square_avg_ : list of ndarray or None
        One buffer per parameter, same shapes, lazily allocated to zeros
        on the first :meth:`step` call. ``None`` before the first call.

    Raises
    ------
    ValueError
        If ``lr <= 0``, ``beta`` is not in ``(0, 1)``, or ``eps <= 0``.

    """

    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8) -> None:
        """See the class docstring for parameter descriptions."""
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 < beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.square_avg_: list[FloatArray] | None = None

    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        """Apply one update, mutating each array in ``params`` in place.

        Parameters
        ----------
        params : list of ndarray
            Parameters to update, any shapes.
        grads : list of ndarray
            Gradients, one per entry of ``params``, same shapes.

        """
        if self.square_avg_ is None:
            self.square_avg_ = [np.zeros_like(p) for p in params]  # s_0 = 0

        for p, g, s in zip(params, grads, self.square_avg_, strict=True):
            s *= self.beta
            s += (1 - self.beta) * g**2  # s = βs + (1-β)g²
            p -= self.lr * g / (np.sqrt(s) + self.eps)  # θ -= η g/(√s + ε)
