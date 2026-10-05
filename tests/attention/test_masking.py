"""Masks (docs/derivations/attention.md section 3)."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.attention import causal_mask, padding_mask


def test_causal_mask_is_lower_triangular():
    np.testing.assert_array_equal(
        causal_mask(3, 3),
        [[True, False, False], [True, True, False], [True, True, True]],
    )


def test_causal_mask_rectangular():
    assert causal_mask(2, 4).shape == (2, 4)
    np.testing.assert_array_equal(causal_mask(2, 4)[1], [True, True, False, False])


def test_padding_mask_blocks_positions_past_length():
    m = padding_mask([2, 4, 0], t_k=4)
    assert m.shape == (3, 1, 4)
    np.testing.assert_array_equal(m[0, 0], [True, True, False, False])
    np.testing.assert_array_equal(m[1, 0], [True, True, True, True])
    np.testing.assert_array_equal(m[2, 0], [False] * 4)


@pytest.mark.parametrize("lengths", [[5], [-1], [[1, 2]]])
def test_padding_mask_rejects_bad_lengths(lengths):
    with pytest.raises(ValueError):
        padding_mask(lengths, t_k=4)
