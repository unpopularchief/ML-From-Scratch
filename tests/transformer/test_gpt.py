"""Tiny GPT shape, gradient, masking, and training contracts."""

import numpy as np
import pytest

from scratchgrad.autograd import Tensor
from scratchgrad.autograd.layers import Embedding
from scratchgrad.autograd.optim import SGD
from scratchgrad.transformer import TinyGPT
from tests.helpers.gradcheck import gradient_check


def test_repeated_array_indices_scatter_add_gradients():
    table = Tensor(np.arange(12.0).reshape(4, 3), requires_grad=True)
    ids = np.array([[1, 2, 1], [3, 1, 2]])
    weights = np.arange(18.0).reshape(2, 3, 3)
    (table[ids] * weights).sum().backward()
    expected = np.zeros_like(table.data)
    expected[1] = weights[0, 0] + weights[0, 2] + weights[1, 1]
    expected[2] = weights[0, 1] + weights[1, 2]
    expected[3] = weights[1, 0]
    np.testing.assert_array_equal(table.grad, expected)
    gradient_check(lambda w: float((w[ids] * weights).sum()), table.grad, table.data)


def test_embedding_rejects_noninteger_indices():
    embedding = Embedding(5, 3, random_state=0)
    with pytest.raises(TypeError, match="integers"):
        embedding(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="out of range"):
        embedding(np.array([-1, 2]))


def test_tiny_gpt_shapes_gradients_and_parameter_update():
    model = TinyGPT(5, 4, 8, 2, 1, 16, random_state=0)
    ids = np.array([[1, 2, 1, 3], [0, 1, 4, 1]])
    logits = model(ids)
    assert logits.shape == (2, 4, 5)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    before = model.token_embedding.weight.data.copy()
    SGD(model.parameters(), lr=0.01).step()
    assert not np.array_equal(model.token_embedding.weight.data, before)


def test_tiny_gpt_is_causal():
    model = TinyGPT(5, 4, 8, 2, 1, 16, random_state=0).eval()
    first = model(np.array([[0, 1, 2, 3]])).data
    changed_future = model(np.array([[0, 1, 4, 4]])).data
    np.testing.assert_allclose(first[:, :2], changed_future[:, :2], atol=1e-12)


def test_tiny_gpt_rejects_non_sequence_input():
    model = TinyGPT(5, 4, 8, 2, 1, 16, random_state=0)
    with pytest.raises(ValueError, match="shape"):
        model(np.array([0, 1, 2]))
