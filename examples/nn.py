"""Training a hand-wired MLP with scratchgrad.nn and scratchgrad.optim.

No ``Sequential``/trainer yet (both separate, later ROADMAP lines) -- this
wires ``Linear -> BatchNorm1d -> ReLU -> Dropout -> Linear ->
BCEWithLogitsLoss`` by hand: each ``forward`` called in order, each
``backward`` called in reverse, and every layer's ``parameters()``/
``grads()`` collected into the flat lists ``optim.Adam.step`` expects. This
is the M3-so-far punchline: the hand-derived backward passes from ``nn/``
and the generic update rule from ``optim/`` (PR #20) working together
end-to-end on a real, non-linearly-separable dataset (docs/derivations/
nn.md section 6, docs/derivations/dropout_batchnorm.md section 4).

Every accuracy/loss checkpoint below switches every stateful layer to
``.eval()`` first and back to ``.train()`` afterward -- ``BatchNorm1d``
normalizes by its running stats rather than the current (evaluation) batch,
and ``Dropout`` stops zeroing units, in eval mode. Skipping that switch
would silently evaluate through a still-randomly-masked, still
current-batch-normalized network -- exactly the bug the ``training``/
``eval`` flag exists to make impossible to forget.

Run:
    uv run python examples/nn.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import make_moons
from scratchgrad.metrics import accuracy_score
from scratchgrad.nn import BatchNorm1d, BCEWithLogitsLoss, Dropout, Linear, Module, ReLU
from scratchgrad.optim import Adam

N_EPOCHS = 500
HIDDEN_UNITS = 16
DROPOUT_P = 0.2


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
    """Train an MLP with BatchNorm1d/Dropout on make_moons; report the accuracy gain."""
    X, y = make_moons(n_samples=300, noise=0.2, random_state=0)
    y = y.astype(np.float64)

    linear1 = Linear(2, HIDDEN_UNITS, random_state=0)
    bn = BatchNorm1d(HIDDEN_UNITS)
    relu = ReLU()
    dropout = Dropout(p=DROPOUT_P, random_state=0)
    linear2 = Linear(HIDDEN_UNITS, 1, random_state=1)
    layers = [linear1, bn, relu, dropout, linear2]
    loss_fn = BCEWithLogitsLoss()
    opt = Adam(lr=0.05)

    def evaluate(x: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
        for layer in layers:
            layer.eval()
        logits = forward(layers, x).ravel()
        loss = loss_fn.forward(logits, y_true)
        acc = accuracy_score(y_true, (logits > 0).astype(np.float64))
        for layer in layers:
            layer.train()
        return loss, acc

    start_loss, start_acc = evaluate(X, y)

    for _ in range(N_EPOCHS):
        logits = forward(layers, X).ravel()
        loss_fn.forward(logits, y)
        grad_logits = loss_fn.backward().reshape(-1, 1)
        backward(layers, grad_logits)

        params = linear1.parameters() + bn.parameters() + linear2.parameters()
        grads = linear1.grads() + bn.grads() + linear2.grads()
        opt.step(params, grads)

    end_loss, end_acc = evaluate(X, y)

    print(
        "make_moons(n_samples=300, noise=0.2), MLP: "
        "2 -> 16 (BatchNorm1d, ReLU, Dropout(0.2)) -> 1"
    )
    print(f"loss:     {start_loss:.4f} -> {end_loss:.4f}")
    print(f"accuracy: {start_acc:.4f} -> {end_acc:.4f}")


if __name__ == "__main__":
    main()
