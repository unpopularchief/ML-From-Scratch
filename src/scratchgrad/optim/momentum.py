r"""SGD with classical ("heavy ball") momentum.

.. math::
    v_{t+1} = \mu v_t - \eta g_t, \qquad \theta_{t+1} = \theta_t + v_{t+1}

Full derivation: ``docs/derivations/optim.md`` section 3.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray


class Momentum:
    r"""Gradient descent with a running velocity term (Polyak, 1964).

    :math:`v_{t+1} = \mu v_t - \eta g_t`, :math:`\theta_{t+1} = \theta_t +
    v_{t+1}`. :math:`v` is an exponentially-decaying running sum of past
    gradients: it accumulates speed along directions where consecutive
    gradients agree in sign and damps oscillation where they don't.
    ``momentum=0`` reduces exactly to plain :class:`~scratchgrad.optim.SGD`.

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

    Examples
    --------
    >>> import numpy as np
    >>> opt = Momentum(lr=0.1, momentum=0.9)
    >>> params = [np.array([1.0])]
    >>> opt.step(params, [np.array([1.0])])
    >>> opt.step(params, [np.array([1.0])])
    >>> params[0]
    array([0.71])

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
        """Apply one update, mutating each array in ``params`` in place.

        Parameters
        ----------
        params : list of ndarray
            Parameters to update, any shapes.
        grads : list of ndarray
            Gradients, one per entry of ``params``, same shapes.

        """
        if self.velocity_ is None:
            self.velocity_ = [np.zeros_like(p) for p in params]  # v_0 = 0

        for p, g, v in zip(params, grads, self.velocity_, strict=True):
            v *= self.momentum
            v -= self.lr * g  # v = μv - ηg
            p += v  # θ += v
