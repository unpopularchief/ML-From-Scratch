"""Synthetic dataset generators, plus real dataset loaders.

``tiny-shakespeare`` arrives at M7.
"""

from scratchgrad.datasets.generators import make_blobs, make_moons, make_regression
from scratchgrad.datasets.mnist import load_mnist

__all__ = ["load_mnist", "make_blobs", "make_moons", "make_regression"]
