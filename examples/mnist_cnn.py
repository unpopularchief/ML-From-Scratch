"""Training a small CNN on real MNIST with scratchgrad.nn.Trainer.

Run:
    uv run python examples/mnist_cnn.py
"""

from __future__ import annotations

import numpy as np

from scratchgrad.datasets import load_mnist
from scratchgrad.metrics import accuracy_score
from scratchgrad.nn import (
    Conv2d,
    CrossEntropyLoss,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Trainer,
)
from scratchgrad.optim import Adam
from scratchgrad.preprocessing import OneHotEncoder

EPOCHS = 3
BATCH_SIZE = 128


def accuracy_metric(logits: np.ndarray, y_true_one_hot: np.ndarray) -> float:
    """Return accuracy from logits and one-hot MNIST labels."""
    return accuracy_score(np.argmax(y_true_one_hot, axis=1), np.argmax(logits, axis=1))


def main() -> None:
    """Train Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear on MNIST."""
    X_train, y_train, X_test, y_test = load_mnist()
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    encoder = OneHotEncoder()
    y_train_oh = encoder.fit_transform(y_train)
    y_test_oh = encoder.transform(y_test)

    layers = [
        Conv2d(1, 8, kernel_size=3, padding=1, random_state=0),
        ReLU(),
        MaxPool2d(2),
        Conv2d(8, 16, kernel_size=3, padding=1, random_state=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(16 * 7 * 7, 10, random_state=2),
    ]
    trainer = Trainer(layers, CrossEntropyLoss(), Adam(lr=1e-3))
    print("MNIST CNN: 1x28x28 -> 8x14x14 -> 16x7x7 -> 10")
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
