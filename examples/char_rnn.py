"""Train a character-level RNN on tiny Shakespeare with manual BPTT.

Run:
    uv run python examples/char_rnn.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import load_tiny_shakespeare
from scratchgrad.nn import RNN, CrossEntropyLoss, Linear
from scratchgrad.optim import Adam

HIDDEN_SIZE = 64
SEQUENCE_LENGTH = 32
BATCH_SIZE = 32
STEPS = 2_000


def make_batch(
    encoded: np.ndarray, vocabulary_size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample contexts and their next characters as one-hot arrays."""
    starts = rng.integers(0, len(encoded) - SEQUENCE_LENGTH, size=BATCH_SIZE)
    offsets = np.arange(SEQUENCE_LENGTH)
    context = encoded[starts[:, None] + offsets]
    target = encoded[starts + SEQUENCE_LENGTH]
    eye = np.eye(vocabulary_size)
    return eye[context], eye[target]


def generate(
    rnn: RNN, head: Linear, chars: np.ndarray, char_to_index: dict[str, int]
) -> str:
    """Greedily continue a short seed, reusing the RNN's batch-first API."""
    generated = list("ROMEO: ")
    for _ in range(300):
        indices = [char_to_index[char] for char in generated[-SEQUENCE_LENGTH:]]
        x = np.eye(len(chars))[indices][None, :, :]
        logits = head.forward(rnn.forward(x)[:, -1])
        generated.append(chars[np.argmax(logits[0])])
    return "".join(generated)


def main() -> None:
    """Train next-character prediction from final RNN states and print a sample."""
    text = load_tiny_shakespeare()
    chars = np.array(sorted(set(text)))
    char_to_index = {char: index for index, char in enumerate(chars)}
    encoded = np.array([char_to_index[char] for char in text], dtype=np.int64)
    rnn = RNN(len(chars), HIDDEN_SIZE, random_state=0)
    head = Linear(HIDDEN_SIZE, len(chars), random_state=1)
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=3e-3)
    rng = np.random.default_rng(0)

    for step in range(1, STEPS + 1):
        x, y = make_batch(encoded, len(chars), rng)
        hidden = rnn.forward(x)
        loss = loss_fn.forward(head.forward(hidden[:, -1]), y)
        grad_h_n = head.backward(loss_fn.backward())
        rnn.backward(np.zeros_like(hidden), grad_h_n=grad_h_n)
        optimizer.step(rnn.parameters() + head.parameters(), rnn.grads() + head.grads())
        if step % 200 == 0:
            print(f"step {step:4d}: loss={loss:.4f}")

    print("\nGenerated text:\n")
    print(generate(rnn, head, chars, char_to_index))


if __name__ == "__main__":
    main()
