"""Naive Bayes classifiers.

One file per algorithm. ``MultinomialNB`` / ``BernoulliNB`` are likely later
additions; only what is implemented is exported.
"""

from scratchgrad.naive_bayes.gaussian_nb import GaussianNB

__all__ = ["GaussianNB"]
