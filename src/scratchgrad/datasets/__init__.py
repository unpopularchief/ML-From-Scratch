"""Synthetic dataset generators, plus real dataset loaders."""

from scratchgrad.datasets.generators import make_blobs, make_moons, make_regression
from scratchgrad.datasets.mnist import load_mnist
from scratchgrad.datasets.shakespeare import load_tiny_shakespeare

__all__ = [
    "load_mnist",
    "load_tiny_shakespeare",
    "make_blobs",
    "make_moons",
    "make_regression",
]
