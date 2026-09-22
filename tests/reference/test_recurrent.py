"""PyTorch parity tests for recurrent cells and sequence-level BPTT."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import LSTM, RNN, LSTMCell, RNNCell

pytestmark = pytest.mark.reference


def _copy_parameters(ours: object, theirs: object, torch: object) -> None:
    """Copy this project's combined-bias parameterization into PyTorch."""
    suffix = "_l0" if hasattr(theirs, "weight_ih_l0") else ""
    with torch.no_grad():
        getattr(theirs, f"weight_ih{suffix}").copy_(torch.tensor(ours.W_ih.T))
        getattr(theirs, f"weight_hh{suffix}").copy_(torch.tensor(ours.W_hh.T))
        getattr(theirs, f"bias_ih{suffix}").copy_(torch.tensor(ours.b))
        getattr(theirs, f"bias_hh{suffix}").zero_()


def _assert_parameter_grads(ours: object, theirs: object) -> None:
    """Compare transposed matrix gradients and the non-redundant bias."""
    suffix = "_l0" if hasattr(theirs, "weight_ih_l0") else ""
    np.testing.assert_allclose(
        ours.grads()[0],
        getattr(theirs, f"weight_ih{suffix}").grad.numpy().T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ours.grads()[1],
        getattr(theirs, f"weight_hh{suffix}").grad.numpy().T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        ours.grads()[2],
        getattr(theirs, f"bias_ih{suffix}").grad.numpy(),
        atol=1e-10,
    )


def test_rnn_cell_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 3))
    h_prev = rng.standard_normal((2, 4))
    grad_h = rng.standard_normal((2, 4))

    ours = RNNCell(3, 4, random_state=1)
    y_ours = ours.forward(x, h_prev)
    dx_ours, dh_ours = ours.backward(grad_h)

    theirs = torch.nn.RNNCell(3, 4, nonlinearity="tanh", dtype=torch.float64)
    _copy_parameters(ours, theirs, torch)
    x_t = torch.tensor(x, requires_grad=True)
    h_t = torch.tensor(h_prev, requires_grad=True)
    y_theirs = theirs(x_t, h_t)
    y_theirs.backward(torch.tensor(grad_h))

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(dh_ours, h_t.grad.numpy(), atol=1e-10)
    _assert_parameter_grads(ours, theirs)


def test_rnn_sequence_and_bptt_match_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 3))
    h0 = rng.standard_normal((2, 4))

    ours = RNN(3, 4, random_state=3)
    y_ours = ours.forward(x, h0)
    grad_output = rng.standard_normal(y_ours.shape)
    grad_h_n = rng.standard_normal((2, 4))
    dx_ours = ours.backward(grad_output, grad_h_n)

    theirs = torch.nn.RNN(
        3, 4, nonlinearity="tanh", batch_first=True, dtype=torch.float64
    )
    _copy_parameters(ours, theirs, torch)
    x_t = torch.tensor(x, requires_grad=True)
    h0_t = torch.tensor(h0[None], requires_grad=True)
    y_theirs, h_n_theirs = theirs(x_t, h0_t)
    torch.autograd.backward(
        (y_theirs, h_n_theirs),
        (torch.tensor(grad_output), torch.tensor(grad_h_n[None])),
    )

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.h_n, h_n_theirs.detach().numpy()[0], atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.grad_h0, h0_t.grad.numpy()[0], atol=1e-10)
    _assert_parameter_grads(ours, theirs)


def test_lstm_cell_matches_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, 3))
    h_prev = rng.standard_normal((2, 4))
    c_prev = rng.standard_normal((2, 4))
    grad_h = rng.standard_normal((2, 4))
    grad_c = rng.standard_normal((2, 4))

    ours = LSTMCell(3, 4, random_state=5)
    h_ours, c_ours = ours.forward(x, (h_prev, c_prev))
    dx_ours, dh_ours, dc_ours = ours.backward(grad_h, grad_c)

    theirs = torch.nn.LSTMCell(3, 4, dtype=torch.float64)
    _copy_parameters(ours, theirs, torch)
    x_t = torch.tensor(x, requires_grad=True)
    h_t = torch.tensor(h_prev, requires_grad=True)
    c_t = torch.tensor(c_prev, requires_grad=True)
    h_theirs, c_theirs = theirs(x_t, (h_t, c_t))
    torch.autograd.backward(
        (h_theirs, c_theirs), (torch.tensor(grad_h), torch.tensor(grad_c))
    )

    np.testing.assert_allclose(h_ours, h_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(c_ours, c_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(dh_ours, h_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(dc_ours, c_t.grad.numpy(), atol=1e-10)
    _assert_parameter_grads(ours, theirs)


def test_lstm_sequence_and_bptt_match_torch() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(6)
    x = rng.standard_normal((2, 3, 3))
    h0 = rng.standard_normal((2, 4))
    c0 = rng.standard_normal((2, 4))

    ours = LSTM(3, 4, random_state=7)
    y_ours = ours.forward(x, (h0, c0))
    grad_output = rng.standard_normal(y_ours.shape)
    grad_h_n = rng.standard_normal((2, 4))
    grad_c_n = rng.standard_normal((2, 4))
    dx_ours = ours.backward(grad_output, grad_h_n, grad_c_n)

    theirs = torch.nn.LSTM(3, 4, batch_first=True, dtype=torch.float64)
    _copy_parameters(ours, theirs, torch)
    x_t = torch.tensor(x, requires_grad=True)
    h0_t = torch.tensor(h0[None], requires_grad=True)
    c0_t = torch.tensor(c0[None], requires_grad=True)
    y_theirs, (h_n_theirs, c_n_theirs) = theirs(x_t, (h0_t, c0_t))
    torch.autograd.backward(
        (y_theirs, h_n_theirs, c_n_theirs),
        (
            torch.tensor(grad_output),
            torch.tensor(grad_h_n[None]),
            torch.tensor(grad_c_n[None]),
        ),
    )

    np.testing.assert_allclose(y_ours, y_theirs.detach().numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.h_n, h_n_theirs.detach().numpy()[0], atol=1e-10)
    np.testing.assert_allclose(ours.c_n, c_n_theirs.detach().numpy()[0], atol=1e-10)
    np.testing.assert_allclose(dx_ours, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.grad_h0, h0_t.grad.numpy()[0], atol=1e-10)
    np.testing.assert_allclose(ours.grad_c0, c0_t.grad.numpy()[0], atol=1e-10)
    _assert_parameter_grads(ours, theirs)
