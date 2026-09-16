r"""Plain stochastic gradient descent.

.. math::
    \theta_{t+1} = \theta_t - \eta\, g_t

Full derivation: ``docs/derivations/optim.md`` section 2.
"""

from __future__ import annotations

from scratchgrad.typing import FloatArray


class SGD:
    r"""Plain (momentum-free) gradient descent.

    :math:`\theta_{t+1} = \theta_t - \eta\, g_t`. "Stochastic" describes how
    the caller computes :math:`g_t` (full-batch or minibatch); ``step``
    itself is the same update either way.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate :math:`\eta`. Must be positive.

    Raises
    ------
    ValueError
        If ``lr <= 0``.

    Examples
    --------
    >>> import numpy as np
    >>> opt = SGD(lr=0.1)
    >>> params = [np.array([1.0, 2.0])]
    >>> opt.step(params, [np.array([1.0, 1.0])])
    >>> params[0]
    array([0.9, 1.9])

    """

    def __init__(self, lr: float = 0.01) -> None:
        """See the class docstring for parameter descriptions."""
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.lr = lr

    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        """Apply one update, mutating each array in ``params`` in place.

        Parameters
        ----------
        params : list of ndarray
            Parameters to update, any shapes.
        grads : list of ndarray
            Gradients, one per entry of ``params``, same shapes.

        """
        for p, g in zip(params, grads, strict=True):
            p -= self.lr * g  # θ -= η g
