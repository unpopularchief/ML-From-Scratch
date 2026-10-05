"""Train a tiny character GPT on tiny Shakespeare and sample its text.

Run:
    uv run python examples/tiny_gpt.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scratchgrad.autograd import Tensor
from scratchgrad.autograd.functional import cross_entropy
from scratchgrad.autograd.optim import Adam
from scratchgrad.datasets import load_tiny_shakespeare
from scratchgrad.transformer import CharTokenizer, TinyGPT

CONTEXT_LENGTH = 32
BATCH_SIZE = 8
STEPS = 2_000
CHECKPOINT_PATH = Path(".cache/tiny-gpt/model.npz")


def make_batch(
    ids: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample contiguous input windows and their one-step-shifted targets."""
    starts = rng.integers(0, len(ids) - CONTEXT_LENGTH, size=BATCH_SIZE)
    positions = starts[:, None] + np.arange(CONTEXT_LENGTH)
    return ids[positions], ids[positions + 1]


def batch_loss(model: TinyGPT, x: np.ndarray, y: np.ndarray) -> Tensor:
    """Mean next-character cross-entropy over all batch positions."""
    logits = model(x).reshape(-1, model.head.b.shape[0])
    targets = Tensor(np.eye(model.head.b.shape[0])[y.reshape(-1)])
    return cross_entropy(logits, targets)


def sample(
    model: TinyGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    rng: np.random.Generator,
    length: int = 300,
) -> str:
    """Draw successive characters from the final position's probabilities."""
    ids = tokenizer.encode(prompt)
    for _ in range(length):
        context = np.array(ids[-CONTEXT_LENGTH:], dtype=np.int64)[None, :]
        logits = model(context).data[0, -1]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        ids.append(int(rng.choice(tokenizer.vocab_size, p=probabilities)))
    return tokenizer.decode(ids)


def main() -> None:
    """Train on the first 90% of the corpus; report held-out loss and a sample."""
    text = load_tiny_shakespeare()
    split = int(0.9 * len(text))
    tokenizer = CharTokenizer().fit(text)
    train_ids = np.array(tokenizer.encode(text[:split]), dtype=np.int64)
    val_ids = np.array(tokenizer.encode(text[split:]), dtype=np.int64)

    model = TinyGPT(
        tokenizer.vocab_size,
        CONTEXT_LENGTH,
        d_model=32,
        num_heads=4,
        num_layers=1,
        d_ff=128,
        random_state=0,
    )
    optimizer = Adam(model.parameters(), lr=3e-3)
    train_rng = np.random.default_rng(1)
    val_rng = np.random.default_rng(2)

    for step in range(1, STEPS + 1):
        x, y = make_batch(train_ids, train_rng)
        optimizer.zero_grad()
        loss = batch_loss(model, x, y)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0:
            model.eval()
            val_losses = [
                float(batch_loss(model, *make_batch(val_ids, val_rng)).data)
                for _ in range(4)
            ]
            train_loss = float(loss.data)
            val_loss = float(np.mean(val_losses))
            print(f"step {step:4d}: train={train_loss:.4f} val={val_loss:.4f}")
            model.train()

    model.eval()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CHECKPOINT_PATH,
        vocabulary=np.array(tokenizer.vocabulary),
        **{f"param_{i}": p.data for i, p in enumerate(model.parameters())},
    )
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print("\nGenerated text:\n")
    print(sample(model, tokenizer, "ROMEO: ", np.random.default_rng(3)))


if __name__ == "__main__":
    main()
