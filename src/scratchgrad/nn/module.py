"""Module: the shared forward/backward/parameters contract for nn layers.

Full derivation: docs/derivations/nn.md section 1.
"""

from __future__ import annotations

from scratchgrad.typing import FloatArray


class Module:
    """Base class for a differentiable layer or activation.

    Not an ABC and not a subclass of ``scratchgrad.base.Estimator`` --
    ``docs/conventions.md`` defines "estimator" as "has ``fit``", and a
    layer has neither ``fit`` nor ``predict``.

    Subclasses implement ``forward``/``backward``; ``parameters()``/
    ``grads()`` default to empty lists, correct as-is for every
    parameterless subclass (every activation in this package).
    """

    def forward(self, x: FloatArray) -> FloatArray:
        """Compute ``y = f(x)``, caching whatever ``backward`` needs."""
        raise NotImplementedError

    def backward(self, grad_output: FloatArray) -> FloatArray:
        """Compute ``dL/dx`` from the upstream gradient ``dL/dy``.

        For a parameterized subclass, also computes and stores
        ``dL/d(each parameter)``, retrievable via :meth:`grads`.
        """
        raise NotImplementedError

    def parameters(self) -> list[FloatArray]:
        """Learnable parameters, in the same order as :meth:`grads`."""
        return []

    def grads(self) -> list[FloatArray]:
        """Gradients from the most recent :meth:`backward` call.

        Same order and shapes as :meth:`parameters` -- directly
        consumable by ``optim.step(module.parameters(), module.grads())``.
        """
        return []
