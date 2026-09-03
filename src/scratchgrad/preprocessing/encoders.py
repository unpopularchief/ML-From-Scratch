"""Categorical encoding transformers."""

from __future__ import annotations

import numpy as np

from scratchgrad.base import Estimator
from scratchgrad.typing import FloatArray, IntArray, TargetVector
from scratchgrad.utils.validation import check_is_fitted


class OneHotEncoder(Estimator):
    """Encode integer-labeled categories as one-hot vectors.

    Each category :math:`c` becomes a length-``n_categories_`` vector with
    a 1 in the position for :math:`c` and 0 elsewhere, so that categorical
    labels can be fed to algorithms (like linear regression) that would
    otherwise treat an arbitrary integer *encoding* as having an ordering
    or magnitude it doesn't really have.

    Attributes
    ----------
    categories_ : ndarray of shape (n_categories,)
        Sorted unique values seen by ``fit``. Column ``j`` of the encoded
        output corresponds to ``categories_[j]``.

    """

    def fit(self, y: TargetVector) -> OneHotEncoder:
        """Learn the set of categories from ``y``.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Integer (or integer-valued) category labels.

        Returns
        -------
        self

        """
        self.categories_ = np.unique(np.asarray(y))
        return self

    def transform(self, y: TargetVector) -> FloatArray:
        """One-hot encode ``y`` using the categories learned by ``fit``.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Category labels to encode.

        Returns
        -------
        ndarray of shape (n_samples, n_categories_)

        Raises
        ------
        ValueError
            If ``y`` contains a category not seen during ``fit``.

        """
        check_is_fitted(self, "categories_")
        y = np.asarray(y)
        unseen = np.setdiff1d(np.unique(y), self.categories_)
        if unseen.size > 0:
            raise ValueError(f"Unseen categories during transform: {unseen.tolist()}.")

        # category -> column index, e.g. {2: 0, 5: 1, 9: 2} for categories_ [2, 5, 9]
        category_to_column: dict[object, int] = {
            category: j for j, category in enumerate(self.categories_)
        }
        column_indices: IntArray = np.array(
            [category_to_column[label] for label in y], dtype=np.int64
        )
        one_hot = np.zeros((len(y), len(self.categories_)), dtype=np.float64)
        one_hot[np.arange(len(y)), column_indices] = 1.0
        return one_hot

    def fit_transform(self, y: TargetVector) -> FloatArray:
        """Equivalent to ``fit(y).transform(y)``."""
        return self.fit(y).transform(y)

    def inverse_transform(self, one_hot: FloatArray) -> TargetVector:
        """Map one-hot rows back to their original category labels."""
        check_is_fitted(self, "categories_")
        column_indices = np.argmax(np.asarray(one_hot), axis=1)
        return self.categories_[column_indices]
