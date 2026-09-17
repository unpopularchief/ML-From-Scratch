"""Training an MLP on real MNIST with scratchgrad.nn.Trainer.

The M3 capstone: every hand-derived piece this milestone built --
``Linear``, ``BatchNorm1d``, ``ReLU``, ``Dropout``, ``CrossEntropyLoss``,
``optim.Adam`` -- composed into one network and trained with
``Trainer.fit``'s minibatch loop rather than the small full-batch,
hand-written loop ``examples/nn.py`` still uses. Downloads MNIST into a
gitignored ``.cache/mnist/`` directory on first run (see
``scratchgrad.datasets.load_mnist``); every later run reuses the cache.

Run:
    uv run python examples/mnist.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import load_mnist
from scratchgrad.metrics import accuracy_score
from scratchgrad.nn import BatchNorm1d, CrossEntropyLoss, Dropout, Linear, ReLU, Trainer
from scratchgrad.optim import Adam
from scratchgrad.preprocessing import OneHotEncoder

EPOCHS = 15
BATCH_SIZE = 128
HIDDEN_1 = 256
HIDDEN_2 = 128
DROPOUT_P = 0.2


def accuracy_metric(logits: np.ndarray, y_true_one_hot: np.ndarray) -> float:
    """Accuracy from raw logits and one-hot targets (both ``(n, 10)``)."""
    y_pred = np.argmax(logits, axis=1)
    y_true = np.argmax(y_true_one_hot, axis=1)
    return accuracy_score(y_true, y_pred)


def main() -> None:
    """Train a 784 -> 256 -> 128 -> 10 MLP on MNIST and report test accuracy."""
    X_train, y_train, X_test, y_test = load_mnist()

    encoder = OneHotEncoder()
    y_train_oh = encoder.fit_transform(y_train)
    y_test_oh = encoder.transform(y_test)

    layers = [
        Linear(784, HIDDEN_1, random_state=0),
        BatchNorm1d(HIDDEN_1),
        ReLU(),
        Dropout(p=DROPOUT_P, random_state=0),
        Linear(HIDDEN_1, HIDDEN_2, random_state=1),
        BatchNorm1d(HIDDEN_2),
        ReLU(),
        Dropout(p=DROPOUT_P, random_state=1),
        Linear(HIDDEN_2, 10, random_state=2),
    ]
    trainer = Trainer(layers, CrossEntropyLoss(), Adam(lr=1e-3))

    print(
        f"MNIST: {X_train.shape[0]} train / {X_test.shape[0]} test, 784 -> "
        f"{HIDDEN_1} -> {HIDDEN_2} -> 10 (BatchNorm1d, ReLU, Dropout({DROPOUT_P}))"
    )

    history = trainer.fit(
        X_train,
        y_train_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        X_val=X_test,
        y_val=y_test_oh,
        metric=accuracy_metric,
        random_state=0,
        verbose=True,
    )

    print(f"final train accuracy: {history['train_metric'][-1]:.4f}")
    print(f"final test accuracy:  {history['val_metric'][-1]:.4f}")


if __name__ == "__main__":
    main()
