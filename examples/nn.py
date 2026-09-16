"""Training a hand-wired MLP with scratchgrad.nn and scratchgrad.optim.

No ``Sequential``/trainer yet (both separate, later ROADMAP lines) -- this
wires ``Linear -> ReLU -> Linear -> BCEWithLogitsLoss`` by hand: each
``forward`` called in order, each ``backward`` called in reverse, and
every layer's ``parameters()``/``grads()`` collected into the flat lists
``optim.Adam.step`` expects. This is the M3-so-far punchline: the
hand-derived backward passes from ``nn/`` and the generic update rule
from ``optim/`` (PR #20) working together end-to-end on a real,
non-linearly-separable dataset (docs/derivations/nn.md section 6).

Run:
    uv run python examples/nn.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import make_moons
from scratchgrad.metrics import accuracy_score
from scratchgrad.nn import BCEWithLogitsLoss, Linear, Module, ReLU
from scratchgrad.optim import Adam

N_EPOCHS = 500
HIDDEN_UNITS = 16


def forward(layers: list[Module], x: np.ndarray) -> np.ndarray:
    """Run ``x`` through every layer in order."""
    for layer in layers:
        x = layer.forward(x)
    return x


def backward(layers: list[Module], grad: np.ndarray) -> None:
    """Run ``grad`` through every layer in reverse order."""
    for layer in reversed(layers):
        grad = layer.backward(grad)


def main() -> None:
    """Train a 2-layer MLP on make_moons and report the accuracy gain."""
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    y = y.astype(np.float64)

    linear1 = Linear(2, HIDDEN_UNITS, random_state=0)
    relu = ReLU()
    linear2 = Linear(HIDDEN_UNITS, 1, random_state=1)
    layers = [linear1, relu, linear2]
    loss_fn = BCEWithLogitsLoss()
    opt = Adam(lr=0.05)

    def predict(x: np.ndarray) -> np.ndarray:
        logits = forward(layers, x).ravel()
        return (logits > 0).astype(np.float64)

    start_loss = loss_fn.forward(forward(layers, X).ravel(), y)
    start_acc = accuracy_score(y, predict(X))

    for _ in range(N_EPOCHS):
        logits = forward(layers, X).ravel()
        loss_fn.forward(logits, y)
        grad_logits = loss_fn.backward().reshape(-1, 1)
        backward(layers, grad_logits)

        params = linear1.parameters() + linear2.parameters()
        grads = linear1.grads() + linear2.grads()
        opt.step(params, grads)

    end_loss = loss_fn.forward(forward(layers, X).ravel(), y)
    end_acc = accuracy_score(y, predict(X))

    print("make_moons(n_samples=300, noise=0.2), MLP: 2 -> 16 (ReLU) -> 1")
    print(f"loss:     {start_loss:.4f} -> {end_loss:.4f}")
    print(f"accuracy: {start_acc:.4f} -> {end_acc:.4f}")


if __name__ == "__main__":
    main()
