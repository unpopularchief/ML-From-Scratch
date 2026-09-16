r"""Adam — bias-corrected running mean and variance of the gradient.

.. math::
    m_{t+1} = \beta_1 m_t + (1-\beta_1) g_t, \qquad
    s_{t+1} = \beta_2 s_t + (1-\beta_2) g_t^2

    \hat m = \frac{m_{t+1}}{1-\beta_1^{t+1}}, \qquad
    \hat s = \frac{s_{t+1}}{1-\beta_2^{t+1}}, \qquad
    \theta_{t+1} = \theta_t - \eta\, \frac{\hat m}{\sqrt{\hat s}+\epsilon}

Full derivation: ``docs/derivations/optim.md`` section 6.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.typing import FloatArray


class Adam:
    r"""Adam (Kingma & Ba, 2015).

    Combines Momentum's running mean of gradients (:math:`m`) with
    RMSprop's running mean of squared gradients (:math:`s`), each
    bias-corrected to counteract the ``0`` initialization (:math:`\hat m,
    \hat s`, dominant only in the first few steps).

    Parameters
    ----------
    lr : float, default=0.001
        Learning rate :math:`\eta`. Must be positive.
    beta1 : float, default=0.9
        Decay :math:`\beta_1 \in (0, 1)` for the first-moment (mean)
        estimate.
    beta2 : float, default=0.999
        Decay :math:`\beta_2 \in (0, 1)` for the second-moment
        (uncentered variance) estimate.
    eps : float, default=1e-8
        Added to :math:`\sqrt{\hat s}` before dividing. Must be positive.

    Attributes
    ----------
    m_ : list of ndarray or None
        First-moment buffers, one per parameter, same shapes. ``None``
        before the first :meth:`step` call.
    s_ : list of ndarray or None
        Second-moment buffers, one per parameter, same shapes. ``None``
        before the first :meth:`step` call.
    t_ : int
        Number of :meth:`step` calls so far — the bias-correction
        exponent.

    Raises
    ------
    ValueError
        If ``lr <= 0``, ``beta1``/``beta2`` is not in ``(0, 1)``, or
        ``eps <= 0``.

    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if not 0 < beta1 < 1:
            raise ValueError(f"beta1 must be in (0, 1), got {beta1}")
        if not 0 < beta2 < 1:
            raise ValueError(f"beta2 must be in (0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_: list[FloatArray] | None = None
        self.s_: list[FloatArray] | None = None
        self.t_ = 0

    def step(self, params: list[FloatArray], grads: list[FloatArray]) -> None:
        """Apply one update, mutating each array in ``params`` in place.

        Parameters
        ----------
        params : list of ndarray
            Parameters to update, any shapes.
        grads : list of ndarray
            Gradients, one per entry of ``params``, same shapes.

        """
        if self.m_ is None:
            self.m_ = [np.zeros_like(p) for p in params]  # m_0 = 0
            self.s_ = [np.zeros_like(p) for p in params]  # s_0 = 0

        self.t_ += 1
        bias1 = 1 - self.beta1**self.t_  # 1 - β1^t
        bias2 = 1 - self.beta2**self.t_  # 1 - β2^t

        for p, g, m, s in zip(params, grads, self.m_, self.s_, strict=True):
            m *= self.beta1
            m += (1 - self.beta1) * g  # m = β1 m + (1-β1) g
            s *= self.beta2
            s += (1 - self.beta2) * g**2  # s = β2 s + (1-β2) g²
            m_hat = m / bias1  # bias-corrected first moment
            s_hat = s / bias2  # bias-corrected second moment
            p -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)
