"""Hand-derived neural network building blocks: manual backward passes.

Every layer, activation, and loss here implements its own `forward`/
`backward` by hand -- no autograd until M5 (`scratchgrad.autograd`). See
``docs/derivations/nn.md`` for the shared ``Module`` interface and each
component's derivation.
"""

from scratchgrad.nn.activations import ReLU, Sigmoid, Softmax, Tanh
from scratchgrad.nn.init import he_normal, xavier_uniform, zeros
from scratchgrad.nn.layers import BatchNorm1d, Dropout, Linear
from scratchgrad.nn.losses import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from scratchgrad.nn.module import Module

__all__ = [
    "BCEWithLogitsLoss",
    "BatchNorm1d",
    "CrossEntropyLoss",
    "Dropout",
    "Linear",
    "MSELoss",
    "Module",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "he_normal",
    "xavier_uniform",
    "zeros",
]
