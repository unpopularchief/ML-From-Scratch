"""Synthetic dataset generators.

Real dataset loaders (MNIST, tiny-shakespeare) arrive at M3+.
"""

from scratchgrad.datasets.generators import make_blobs, make_moons, make_regression

__all__ = ["make_blobs", "make_moons", "make_regression"]
