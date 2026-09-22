"""PyTorch parity tests for Conv2d, MaxPool2d, and Flatten."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import Conv2d, Flatten, MaxPool2d

pytestmark = pytest.mark.reference


def test_conv2d_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 2, 5, 6))

    ours = Conv2d(
        2,
        3,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 0),
        random_state=0,
    )
    y_ours = ours.forward(x)
    grad_output = rng.standard_normal(y_ours.shape)
    dx_ours = ours.backward(grad_output)

    theirs = torch.nn.Conv2d(
        2,
        3,
        kernel_size=(3, 2),
        stride=(2, 1),
        padding=(1, 0),
        dtype=torch.float64,
    )
    with torch.no_grad():
        theirs.weight.copy_(torch.tensor(ours.W.copy()))
        theirs.bias.copy_(torch.tensor(ours.b.copy()))
    x_t = torch.tensor(x, requires_grad=True)
    y_theirs = theirs(x_t)
    y_theirs.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.grads()[0], theirs.weight.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.grads()[1], theirs.bias.grad.numpy(), atol=1e-10)


def test_maxpool2d_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4, 6))

    ours = MaxPool2d(kernel_size=(2, 3), stride=(1, 2), padding=(0, 1))
    y_ours = ours.forward(x)
    grad_output = rng.standard_normal(y_ours.shape)
    dx_ours = ours.backward(grad_output)

    x_t = torch.tensor(x, requires_grad=True)
    y_theirs = torch.nn.functional.max_pool2d(
        x_t, kernel_size=(2, 3), stride=(1, 2), padding=(0, 1)
    )
    y_theirs.backward(torch.tensor(grad_output))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)


def test_flatten_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3, 4, 5))
    grad_output = rng.standard_normal((2, 60))

    ours = Flatten()
    y_ours = ours.forward(x)
    dx_ours = ours.backward(grad_output)

    x_t = torch.tensor(x, requires_grad=True)
    y_theirs = torch.flatten(x_t, start_dim=1)
    y_theirs.backward(torch.tensor(grad_output))

    np.testing.assert_array_equal(y_ours, y_theirs.detach().numpy())
    np.testing.assert_array_equal(dx_ours, x_t.grad.numpy())
