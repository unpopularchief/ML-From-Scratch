"""Hand-derived neural network building blocks: manual backward passes.

Every layer, activation, and loss here implements its own `forward`/
`backward` by hand -- no autograd until M5 (`scratchgrad.autograd`). See
``docs/derivations/nn.md`` for the shared ``Module`` interface and M3
components; ``docs/derivations/conv_pool.md`` covers M4's CNN layers.
"""

from scratchgrad.nn.activations import ReLU, Sigmoid, Softmax, Tanh
from scratchgrad.nn.init import he_normal, xavier_uniform, zeros
from scratchgrad.nn.layers import (
    BatchNorm1d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
)
from scratchgrad.nn.losses import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from scratchgrad.nn.module import Module
from scratchgrad.nn.trainer import Trainer

__all__ = [
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Flatten",
    "Linear",
    "MaxPool2d",
    "MSELoss",
    "Module",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "Trainer",
    "he_normal",
    "xavier_uniform",
    "zeros",
]
