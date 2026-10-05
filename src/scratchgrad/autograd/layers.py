r"""Layers on ``Tensor``: no hand-written backward, the engine supplies it.

Full derivation: docs/derivations/autograd_layers.md section 1.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from scratchgrad.autograd.functional import layer_norm
from scratchgrad.autograd.tensor import Tensor
from scratchgrad.nn.init import he_normal, xavier_uniform, zeros
from scratchgrad.nn.layers.dropout import _dropout_mask
from scratchgrad.utils.validation import check_random_state

_WEIGHT_INITS = ("he", "xavier", "zeros")


class Module:
    """Minimal base: parameter collection, ``zero_grad`` and train/eval mode.

    Subclasses implement ``forward`` and override :meth:`parameters` if they
    own parameters or child modules.
    """

    training: bool = True

    def forward(self, x: Tensor) -> Tensor:
        """Compute the layer output."""
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def parameters(self) -> list[Tensor]:
        """All trainable Tensors (none by default)."""
        return []

    def zero_grad(self) -> None:
        """Reset every parameter's ``grad``."""
        for p in self.parameters():
            p.zero_grad()

    def train(self) -> Module:
        """Switch to training mode and return ``self``."""
        self.training = True
        return self

    def eval(self) -> Module:
        """Switch to evaluation mode and return ``self``."""
        self.training = False
        return self


class Linear(Module):
    r"""Fully-connected layer, :math:`Y = XW + b`, on Tensors.

    Same arguments and initialization as ``nn.Linear``, so equal seeds give
    equal weights.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Linear(3, 2, random_state=0)
    >>> layer(Tensor(np.ones((4, 3)))).shape
    (4, 2)

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_init: str = "he",
        random_state: int | None = None,
    ) -> None:
        """See ``nn.Linear`` for parameter descriptions."""
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if out_features <= 0:
            raise ValueError(f"out_features must be positive, got {out_features}")
        if weight_init not in _WEIGHT_INITS:
            raise ValueError(
                f"weight_init must be one of {_WEIGHT_INITS}, got {weight_init!r}"
            )
        rng = check_random_state(random_state)
        if weight_init == "he":
            w = he_normal(in_features, out_features, rng)
        elif weight_init == "xavier":
            w = xavier_uniform(in_features, out_features, rng)
        else:
            w = zeros((in_features, out_features))
        self.W = Tensor(w, requires_grad=True)
        self.b = Tensor(zeros((out_features,)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ W + b``."""
        return x @ self.W + self.b

    def parameters(self) -> list[Tensor]:
        """``[W, b]``."""
        return [self.W, self.b]


class LayerNorm(Module):
    """Normalize over the last axis with learned scale ``gamma`` and shift ``beta``.

    Examples
    --------
    >>> import numpy as np
    >>> layer = LayerNorm(4)
    >>> layer(Tensor(np.arange(8.0).reshape(2, 4))).shape
    (2, 4)

    """

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        """``gamma`` starts at one and ``beta`` at zero."""
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        self.eps = eps
        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply :func:`scratchgrad.autograd.functional.layer_norm`."""
        return layer_norm(x, self.gamma, self.beta, self.eps)

    def parameters(self) -> list[Tensor]:
        """``[gamma, beta]``."""
        return [self.gamma, self.beta]


class Dropout(Module):
    """Inverted dropout: ``x * bernoulli(1-p)/(1-p)`` in training, identity in eval.

    Examples
    --------
    >>> import numpy as np
    >>> layer = Dropout(p=0.5, random_state=0).eval()
    >>> layer(Tensor(np.ones(3))).data
    array([1., 1., 1.])

    """

    def __init__(self, p: float = 0.5, random_state: int | None = None) -> None:
        """``p`` in ``[0, 1)``; ``random_state`` seeds the advancing mask stream."""
        if not (0.0 <= p < 1.0):
            raise ValueError(f"p must be in [0, 1), got {p}")
        self.p = p
        self._rng = check_random_state(random_state)

    def forward(self, x: Tensor) -> Tensor:
        """Multiply by a fresh constant mask in training mode, else the identity."""
        if not self.training:
            return x
        return x * _dropout_mask(x.shape, self.p, self._rng)


class Sequential(Module):
    """Apply modules and plain callables (e.g. ``ops.relu``) in order.

    Examples
    --------
    >>> from scratchgrad.autograd import ops
    >>> net = Sequential(Linear(2, 3, random_state=0), ops.relu, Linear(3, 1))
    >>> len(net.parameters())
    4

    """

    def __init__(self, *layers: Callable[[Tensor], Tensor]) -> None:
        """Store ``layers`` in application order."""
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        """Feed ``x`` through every layer."""
        for layer in self.layers:
            x = layer(x)
        return x

    def _modules(self) -> list[Module]:
        return [layer for layer in self.layers if isinstance(layer, Module)]

    def parameters(self) -> list[Tensor]:
        """Parameters of every child module, in order."""
        return [p for m in self._modules() for p in m.parameters()]

    def train(self) -> Module:
        """Switch this module and every child to training mode."""
        super().train()
        for m in self._modules():
            m.train()
        return self

    def eval(self) -> Module:
        """Switch this module and every child to evaluation mode."""
        super().eval()
        for m in self._modules():
            m.eval()
        return self
