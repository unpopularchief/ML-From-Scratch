"""M5 proof test: an MLP on autograd reproduces the hand-derived M3 MLP.

The same network is built twice from the same seeds: once from ``nn`` layers
whose backward passes were derived by hand, once from ``autograd`` layers with
no backward code at all. The test asserts that the loss, every parameter
gradient, and a whole Adam training trajectory agree.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.autograd import Tensor, ops
from scratchgrad.autograd import functional as F
from scratchgrad.autograd.layers import Dropout, Linear, Sequential
from scratchgrad.autograd.optim import Adam as AutogradAdam
from scratchgrad.nn import CrossEntropyLoss, Tanh
from scratchgrad.nn import Dropout as NnDropout
from scratchgrad.nn import Linear as NnLinear
from scratchgrad.nn import ReLU as NnReLU
from scratchgrad.optim import Adam as HandAdam

N, D_IN, H1, H2, K = 32, 5, 8, 6, 3
DROPOUT_P = 0.25
ATOL = 1e-10

rng = np.random.default_rng(0)
X = rng.standard_normal((N, D_IN))
Y = np.eye(K)[rng.integers(0, K, size=N)]


def build_hand() -> list:
    return [
        NnLinear(D_IN, H1, random_state=1),
        NnReLU(),
        NnDropout(p=DROPOUT_P, random_state=2),
        NnLinear(H1, H2, weight_init="xavier", random_state=3),
        Tanh(),
        NnLinear(H2, K, random_state=4),
    ]


def build_autograd() -> Sequential:
    return Sequential(
        Linear(D_IN, H1, random_state=1),
        ops.relu,
        Dropout(p=DROPOUT_P, random_state=2),
        Linear(H1, H2, weight_init="xavier", random_state=3),
        ops.tanh,
        Linear(H2, K, random_state=4),
    )


def hand_step(layers: list, loss_fn: CrossEntropyLoss) -> float:
    """One forward + hand-derived backward pass; return the loss."""
    out = X
    for layer in layers:
        out = layer.forward(out)
    loss = loss_fn.forward(out, Y)
    grad = loss_fn.backward()
    for layer in reversed(layers):
        grad = layer.backward(grad)
    return loss


def hand_params_and_grads(layers: list) -> tuple[list, list]:
    params = [p for layer in layers for p in layer.parameters()]
    grads = [g for layer in layers for g in layer.grads()]
    return params, grads


def test_loss_and_every_gradient_match_hand_derived() -> None:
    hand, loss_fn = build_hand(), CrossEntropyLoss()
    net = build_autograd()

    hand_loss = hand_step(hand, loss_fn)
    loss = F.cross_entropy(net(Tensor(X)), Tensor(Y))
    loss.backward()

    np.testing.assert_allclose(loss.data, hand_loss, atol=ATOL)
    hand_params, hand_grads = hand_params_and_grads(hand)
    assert len(net.parameters()) == len(hand_params) == 6
    for p, w, g in zip(net.parameters(), hand_params, hand_grads, strict=True):
        np.testing.assert_array_equal(p.data, w)  # identical initialization
        np.testing.assert_allclose(p.grad, g, atol=ATOL)


def test_input_gradient_matches_hand_derived() -> None:
    hand, loss_fn = build_hand(), CrossEntropyLoss()
    net = build_autograd()

    out = X
    for layer in hand:
        out = layer.forward(out)
    loss_fn.forward(out, Y)
    grad = loss_fn.backward()
    for layer in reversed(hand):
        grad = layer.backward(grad)

    x = Tensor(X, requires_grad=True)
    F.cross_entropy(net(x), Tensor(Y)).backward()
    np.testing.assert_allclose(x.grad, grad, atol=ATOL)


def test_adam_training_trajectory_matches() -> None:
    hand, loss_fn = build_hand(), CrossEntropyLoss()
    hand_params, _ = hand_params_and_grads(hand)
    hand_opt = HandAdam(lr=0.02)

    net = build_autograd()
    opt = AutogradAdam(net.parameters(), lr=0.02)

    first = last = None
    for _ in range(25):
        hand_loss = hand_step(hand, loss_fn)
        _, hand_grads = hand_params_and_grads(hand)
        hand_opt.step(hand_params, hand_grads)

        opt.zero_grad()
        loss = F.cross_entropy(net(Tensor(X)), Tensor(Y))
        loss.backward()
        opt.step()

        np.testing.assert_allclose(loss.data, hand_loss, atol=ATOL)
        first = hand_loss if first is None else first
        last = hand_loss

    for p, w in zip(net.parameters(), hand_params, strict=True):
        np.testing.assert_allclose(p.data, w, atol=1e-8)
    assert last < first  # the shared trajectory actually learns
