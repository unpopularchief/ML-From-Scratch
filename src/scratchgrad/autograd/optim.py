r"""Optimizers over ``Tensor`` parameters: SGD, Adam, AdamW.

Same update rules as ``scratchgrad.optim`` (docs/derivations/optim.md); the
interface differs: parameters are held by the optimizer and ``step()`` reads
each ``p.grad``, updating ``p.data`` in place with no graph recorded.

Full derivation: docs/derivations/autograd_layers.md section 3.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.autograd.tensor import Tensor


class Optimizer:
    """Holds the parameters and provides ``zero_grad``."""

    def __init__(self, params: list[Tensor], lr: float) -> None:
        """``params`` are Tensors with ``requires_grad=True``; ``lr`` positive."""
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        """Reset every parameter's ``grad``."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Apply one update."""
        raise NotImplementedError


class SGD(Optimizer):
    r""":math:`\theta \leftarrow \theta - \eta g`.

    Examples
    --------
    >>> w = Tensor([1.0, 2.0], requires_grad=True)
    >>> (w * w).sum().backward()
    >>> SGD([w], lr=0.1).step()
    >>> w.data
    array([0.8, 1.6])

    """

    def __init__(self, params: list[Tensor], lr: float = 0.01) -> None:
        """See :class:`Optimizer`."""
        super().__init__(params, lr)

    def step(self) -> None:
        """Update every parameter that received a gradient."""
        for p in self.params:
            if p.grad is not None:
                p.data -= self.lr * p.grad  # θ -= η g


class Adam(Optimizer):
    r"""Adam (Kingma & Ba, 2015); rule as in ``scratchgrad.optim.Adam``."""

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """``beta1``/``beta2`` in ``(0, 1)``; ``eps`` positive."""
        super().__init__(params, lr)
        if not 0 < beta1 < 1:
            raise ValueError(f"beta1 must be in (0, 1), got {beta1}")
        if not 0 < beta2 < 1:
            raise ValueError(f"beta2 must be in (0, 1), got {beta2}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_ = [np.zeros_like(p.data) for p in self.params]  # m_0 = 0
        self.s_ = [np.zeros_like(p.data) for p in self.params]  # s_0 = 0
        self.t_ = 0

    def step(self) -> None:
        """Update every parameter that received a gradient."""
        self.t_ += 1
        bias1 = 1 - self.beta1**self.t_  # 1 - β1^t
        bias2 = 1 - self.beta2**self.t_  # 1 - β2^t
        for p, m, s in zip(self.params, self.m_, self.s_, strict=True):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.beta1
            m += (1 - self.beta1) * g  # m = β1 m + (1-β1) g
            s *= self.beta2
            s += (1 - self.beta2) * g**2  # s = β2 s + (1-β2) g²
            p.data -= self.lr * (m / bias1) / (np.sqrt(s / bias2) + self.eps)


class AdamW(Adam):
    r"""Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Shrinks :math:`\theta \leftarrow \theta (1 - \eta\lambda)` directly, then
    takes the plain Adam step on the unmodified gradient, so the decay is not
    rescaled by :math:`\sqrt{\hat s}`.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        """``weight_decay`` must be non-negative."""
        super().__init__(params, lr, beta1, beta2, eps)
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
        self.weight_decay = weight_decay

    def step(self) -> None:
        """Decay, then Adam-update, every parameter that received a gradient."""
        for p in self.params:
            if p.grad is not None:
                p.data *= 1 - self.lr * self.weight_decay  # θ *= 1 - ηλ
        super().step()
