r"""Nesterov accelerated gradient, reformulated for a current-point gradient.

.. math::
    v_{t+1} = \mu v_t - \eta g_t, \qquad
    \psi_{t+1} = \psi_t - \mu v_t + (1+\mu) v_{t+1}

Full derivation: ``docs/derivations/optim.md`` section 4 — in particular,
*why* this is not the textbook lookahead-gradient formula and what
substitution makes it equivalent to it while only ever needing the
gradient at the point ``step`` is actually called with.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray


class Nesterov:
    r"""Nesterov accelerated gradient (Sutskever et al., 2013 reformulation).

    Textbook NAG evaluates the gradient at a lookahead point :math:`\theta_t
    + \mu v_t`, which the ``step(params, grads)`` interface cannot supply
    (``grads`` is always the gradient at the current ``params``). Sutskever,
    Martens, Dahl & Hinton (2013) show that tracking the *lookahead* point
    itself, :math:`\psi_t = \theta_t + \mu v_t`, as the externally-visible
    parameter makes the recursion solvable using only the gradient at the
    current point:

    .. math::
        v_{t+1} = \mu v_t - \eta g_t, \qquad
        \psi_{t+1} = \psi_t - \mu v_t + (1+\mu) v_{t+1}

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate :math:`\eta`. Must be positive.
    momentum : float, default=0.9
        Decay :math:`\mu \in [0, 1)` for the velocity buffer.

    Attributes
    ----------
    velocity_ : list of ndarray or None
        One buffer per parameter, same shapes, lazily allocated to zeros
        on the first :meth:`step` call. ``None`` before the first call.

    Raises
    ------
    ValueError
        If ``lr <= 0`` or ``momentum`` is not in ``[0, 1)``.

    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        """See the class docstring for parameter descriptions."""
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 <= momentum < 1:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.lr = lr
        self.momentum = momentum
        self.velocity_: list[FloatArray] | None = None

    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        r"""Apply one update, mutating each array in ``params`` in place.

        Parameters
        ----------
        params : list of ndarray
            Parameters to update (tracked as the lookahead point
            :math:`\psi`), any shapes.
        grads : list of ndarray
            Gradients at the current ``params``, one per entry, same shapes.

        """
        if self.velocity_ is None:
            self.velocity_ = [np.zeros_like(p) for p in params]  # v_0 = 0

        for p, g, v in zip(params, grads, self.velocity_, strict=True):
            v_prev = v.copy()  # v_t, needed after v is overwritten below
            v *= self.momentum
            v -= self.lr * g  # v_{t+1} = μv_t - ηg_t
            p -= self.momentum * v_prev
            p += (1 + self.momentum) * v  # ψ_{t+1} = ψ_t - μv_t + (1+μ)v_{t+1}
