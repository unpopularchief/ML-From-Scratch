"""Racing SGD, Momentum, Nesterov, RMSprop, and Adam on a quadratic bowl.

Minimizes ``L(theta) = 1/2 theta^T A theta`` (gradient ``A @ theta``,
minimum at ``theta = 0``) for a deliberately ill-conditioned diagonal ``A``
(condition number 100): one axis is steep, the other shallow. Plain SGD
must use a small learning rate to stay stable on the steep axis, which
makes it slow on the shallow one -- exactly the failure mode Momentum,
Nesterov, RMSprop, and Adam each address differently
(docs/derivations/optim.md sections 3-6).

Run:
    uv run python examples/optim.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.optim import SGD, Adam, Momentum, Nesterov, RMSprop

N_STEPS = 200
A = np.diag([1.0, 100.0])  # condition number 100
THETA0 = np.array([10.0, 1.0])


def loss(theta: np.ndarray) -> float:
    """Evaluate ``1/2 theta^T A theta``."""
    return float(0.5 * theta @ A @ theta)


def race(name: str, opt, lr: float) -> None:
    """Run ``opt`` for ``N_STEPS`` steps from ``THETA0`` and print the loss curve."""
    theta = THETA0.copy()
    losses = [loss(theta)]
    for _ in range(N_STEPS):
        opt.step([theta], [A @ theta])
        losses.append(loss(theta))
    print(f"{name:<10} lr={lr:<6} loss: {losses[0]:.4f} -> {losses[-1]:.6f}")


def main() -> None:
    """Race the five optimizers on the ill-conditioned bowl."""
    print(f"L(theta) = 1/2 theta^T A theta, A = diag(1, 100), theta0 = {THETA0}")
    race("SGD", SGD(lr=0.005), 0.005)
    race("Momentum", Momentum(lr=0.005, momentum=0.9), 0.005)
    race("Nesterov", Nesterov(lr=0.005, momentum=0.9), 0.005)
    race("RMSprop", RMSprop(lr=0.1), 0.1)
    race("Adam", Adam(lr=0.1), 0.1)


if __name__ == "__main__":
    main()
