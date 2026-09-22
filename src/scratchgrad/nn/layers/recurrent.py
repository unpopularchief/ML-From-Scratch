r"""Vanilla RNN and LSTM cells with full backpropagation through time.

Full derivation: docs/derivations/recurrent.md.
"""

from __future__ import annotations

import numpy as np

from scratchgrad.nn.init import xavier_uniform, zeros
from scratchgrad.nn.module import Module
from scratchgrad.typing import FloatArray
from scratchgrad.utils.math import sigmoid
from scratchgrad.utils.validation import check_random_state

_WEIGHT_INITS = ("xavier", "zeros")


def _validate_sizes(input_size: int, hidden_size: int, weight_init: str) -> None:
    """Validate recurrent-layer constructor arguments."""
    if input_size <= 0:
        raise ValueError(f"input_size must be positive, got {input_size}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if weight_init not in _WEIGHT_INITS:
        raise ValueError(
            f"weight_init must be one of {_WEIGHT_INITS}, got {weight_init!r}"
        )


def _initial_state(
    state: FloatArray | None,
    batch_size: int,
    hidden_size: int,
    name: str,
) -> FloatArray:
    """Return a supplied recurrent state or a float64 zero state."""
    if state is None:
        return np.zeros((batch_size, hidden_size), dtype=np.float64)
    expected = (batch_size, hidden_size)
    if state.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {state.shape}")
    return state


def _check_step_input(
    x: FloatArray,
    h_prev: FloatArray,
    input_size: int,
    hidden_size: int,
) -> None:
    """Validate one recurrent timestep's input and previous hidden state."""
    if x.ndim != 2 or x.shape[1] != input_size:
        raise ValueError(f"x must have shape (N, {input_size}), got {x.shape}")
    expected_h = (x.shape[0], hidden_size)
    if h_prev.shape != expected_h:
        raise ValueError(f"h_prev must have shape {expected_h}, got {h_prev.shape}")


def _check_sequence_input(x: FloatArray, input_size: int) -> None:
    """Validate a non-empty batch-first sequence."""
    if x.ndim != 3 or x.shape[2] != input_size:
        raise ValueError(f"x must have shape (N, T, {input_size}), got {x.shape}")
    if x.shape[1] == 0:
        raise ValueError("x must contain at least one timestep")


def _init_weights(
    input_size: int,
    hidden_size: int,
    output_size: int,
    weight_init: str,
    random_state: int | np.random.Generator | None,
) -> tuple[FloatArray, FloatArray]:
    """Initialize the input and recurrent matrices independently."""
    rng = check_random_state(random_state)
    if weight_init == "xavier":
        W_ih = xavier_uniform(input_size, output_size, rng)
        W_hh = xavier_uniform(hidden_size, output_size, rng)
    else:
        W_ih = zeros((input_size, output_size))
        W_hh = zeros((hidden_size, output_size))
    return W_ih, W_hh


def _rnn_cell_forward(
    x: FloatArray,
    h_prev: FloatArray,
    W_ih: FloatArray,
    W_hh: FloatArray,
    b: FloatArray | None,
) -> FloatArray:
    r"""Compute :math:`h = \tanh(xW_{ih} + h_{prev}W_{hh} + b)`."""
    activation = x @ W_ih + h_prev @ W_hh
    if b is not None:
        activation += b
    return np.tanh(activation)


def _rnn_cell_backward(
    x: FloatArray,
    h_prev: FloatArray,
    h: FloatArray,
    W_ih: FloatArray,
    W_hh: FloatArray,
    grad_h: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return ``(dX, dH_prev, dW_ih, dW_hh, db)`` for one RNN step."""
    grad_activation = grad_h * (1.0 - h * h)
    dx = grad_activation @ W_ih.T
    grad_h_prev = grad_activation @ W_hh.T
    dW_ih = x.T @ grad_activation
    dW_hh = h_prev.T @ grad_activation
    db = grad_activation.sum(axis=0)
    return dx, grad_h_prev, dW_ih, dW_hh, db


def _lstm_cell_forward(
    x: FloatArray,
    h_prev: FloatArray,
    c_prev: FloatArray,
    W_ih: FloatArray,
    W_hh: FloatArray,
    b: FloatArray | None,
) -> tuple[FloatArray, FloatArray, tuple[FloatArray, ...]]:
    """Compute one LSTM step in ``(input, forget, candidate, output)`` order."""
    activation = x @ W_ih + h_prev @ W_hh
    if b is not None:
        activation += b
    a_i, a_f, a_g, a_o = np.split(activation, 4, axis=1)
    i = sigmoid(a_i)
    f = sigmoid(a_f)
    g = np.tanh(a_g)
    o = sigmoid(a_o)
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c, (i, f, g, o)


def _lstm_cell_backward(
    x: FloatArray,
    h_prev: FloatArray,
    c_prev: FloatArray,
    c: FloatArray,
    gates: tuple[FloatArray, ...],
    W_ih: FloatArray,
    W_hh: FloatArray,
    grad_h: FloatArray,
    grad_c: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return input/state/parameter gradients for one LSTM step."""
    i, f, g, o = gates
    tanh_c = np.tanh(c)

    grad_o = grad_h * tanh_c
    grad_c_total = grad_c + grad_h * o * (1.0 - tanh_c * tanh_c)
    grad_f = grad_c_total * c_prev
    grad_c_prev = grad_c_total * f
    grad_i = grad_c_total * g
    grad_g = grad_c_total * i

    grad_activation = np.concatenate(
        (
            grad_i * i * (1.0 - i),
            grad_f * f * (1.0 - f),
            grad_g * (1.0 - g * g),
            grad_o * o * (1.0 - o),
        ),
        axis=1,
    )
    dx = grad_activation @ W_ih.T
    grad_h_prev = grad_activation @ W_hh.T
    dW_ih = x.T @ grad_activation
    dW_hh = h_prev.T @ grad_activation
    db = grad_activation.sum(axis=0)
    return dx, grad_h_prev, grad_c_prev, dW_ih, dW_hh, db


class RNNCell(Module):
    r"""One tanh recurrent-neural-network step.

    ``forward(x, h_prev=None)`` computes
    :math:`h=\tanh(xW_{ih}+h_{prev}W_{hh}+b)`. ``backward(grad_h)``
    returns ``(dX, dH_prev)`` and caches the parameter gradients.

    Parameters
    ----------
    input_size : int
        Number of features in one timestep.
    hidden_size : int
        Number of recurrent hidden units.
    bias : bool, default=True
        Whether to add a learnable bias.
    weight_init : {"xavier", "zeros"}, default="xavier"
        Initialization applied independently to both weight matrices.
    random_state : int, numpy.random.Generator, or None, default=None
        Seeds weight initialization.

    Examples
    --------
    >>> import numpy as np
    >>> cell = RNNCell(3, 4, random_state=0)
    >>> cell.forward(np.ones((2, 3))).shape
    (2, 4)

    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        weight_init: str = "xavier",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See the class docstring for parameter descriptions."""
        _validate_sizes(input_size, hidden_size, weight_init)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_init = weight_init
        self.random_state = random_state
        self.W_ih, self.W_hh = _init_weights(
            input_size, hidden_size, hidden_size, weight_init, random_state
        )
        self.b = zeros((hidden_size,)) if bias else None

        self._x: FloatArray | None = None
        self._h_prev: FloatArray | None = None
        self._h: FloatArray | None = None
        self._dW_ih: FloatArray | None = None
        self._dW_hh: FloatArray | None = None
        self._db: FloatArray | None = None

    def forward(self, x: FloatArray, h_prev: FloatArray | None = None) -> FloatArray:
        """Compute one recurrent step, using zeros when ``h_prev`` is omitted."""
        if x.ndim != 2:
            raise ValueError(f"x must have shape (N, {self.input_size}), got {x.shape}")
        h_prev = _initial_state(h_prev, x.shape[0], self.hidden_size, "h_prev")
        _check_step_input(x, h_prev, self.input_size, self.hidden_size)
        self._x, self._h_prev = x, h_prev
        self._h = _rnn_cell_forward(x, h_prev, self.W_ih, self.W_hh, self.b)
        return self._h

    def backward(self, grad_h: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Cache parameter gradients and return ``(dX, dH_prev)``."""
        expected = self._h.shape
        if grad_h.shape != expected:
            raise ValueError(f"grad_h must have shape {expected}, got {grad_h.shape}")
        dx, grad_h_prev, self._dW_ih, self._dW_hh, db = _rnn_cell_backward(
            self._x, self._h_prev, self._h, self.W_ih, self.W_hh, grad_h
        )
        self._db = db if self.bias else None
        return dx, grad_h_prev

    def parameters(self) -> list[FloatArray]:
        """Return ``[W_ih, W_hh, b]`` when biased, otherwise both weights."""
        params = [self.W_ih, self.W_hh]
        return [*params, self.b] if self.bias else params

    def grads(self) -> list[FloatArray]:
        """Return gradients in the same order as :meth:`parameters`."""
        grads = [self._dW_ih, self._dW_hh]
        return [*grads, self._db] if self.bias else grads


class RNN(RNNCell):
    r"""Batch-first tanh RNN with a full manual BPTT pass.

    Input has shape ``(N, T, input_size)`` and output contains every hidden
    state with shape ``(N, T, hidden_size)``. The final state is available
    as :attr:`h_n`. ``backward`` returns ``dX`` and stores ``grad_h0``.

    An optional ``grad_h_n`` is *additional* to the gradient already supplied
    at the last position of ``grad_output``. It represents a separate loss
    edge attached directly to :attr:`h_n`.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        weight_init: str = "xavier",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See :class:`RNNCell`; this layer unrolls the cell over time."""
        super().__init__(input_size, hidden_size, bias, weight_init, random_state)
        self.h_n: FloatArray | None = None
        self.grad_h0: FloatArray | None = None
        self._hidden: FloatArray | None = None

    def forward(self, x: FloatArray, h0: FloatArray | None = None) -> FloatArray:
        """Unroll the recurrence over a batch-first input sequence."""
        _check_sequence_input(x, self.input_size)
        n, timesteps, _ = x.shape
        h0 = _initial_state(h0, n, self.hidden_size, "h0")
        hidden = np.empty((n, timesteps + 1, self.hidden_size), dtype=np.float64)
        hidden[:, 0] = h0
        for t in range(timesteps):
            hidden[:, t + 1] = _rnn_cell_forward(
                x[:, t], hidden[:, t], self.W_ih, self.W_hh, self.b
            )
        self._x, self._hidden = x, hidden
        self.h_n = hidden[:, -1].copy()
        return hidden[:, 1:]

    def backward(
        self,
        grad_output: FloatArray,
        grad_h_n: FloatArray | None = None,
    ) -> FloatArray:
        """Run full reverse-time BPTT, returning ``dX``."""
        n, timesteps, _ = self._x.shape
        expected = (n, timesteps, self.hidden_size)
        if grad_output.shape != expected:
            raise ValueError(
                f"grad_output must have shape {expected}, got {grad_output.shape}"
            )
        grad_h_future = _initial_state(grad_h_n, n, self.hidden_size, "grad_h_n").copy()
        dx = np.empty_like(self._x, dtype=np.float64)
        self._dW_ih = np.zeros_like(self.W_ih)
        self._dW_hh = np.zeros_like(self.W_hh)
        db = np.zeros(self.hidden_size, dtype=np.float64)

        for t in range(timesteps - 1, -1, -1):
            grad_h = grad_output[:, t] + grad_h_future
            (
                dx[:, t],
                grad_h_future,
                dW_ih_t,
                dW_hh_t,
                db_t,
            ) = _rnn_cell_backward(
                self._x[:, t],
                self._hidden[:, t],
                self._hidden[:, t + 1],
                self.W_ih,
                self.W_hh,
                grad_h,
            )
            self._dW_ih += dW_ih_t
            self._dW_hh += dW_hh_t
            db += db_t

        self.grad_h0 = grad_h_future
        self._db = db if self.bias else None
        return dx


class LSTMCell(Module):
    r"""One long short-term memory step in ``i, f, g, o`` gate order.

    ``forward(x, state=None)`` returns ``(h, c)``. The zero-default ``state``
    is a ``(h_prev, c_prev)`` pair. ``backward(grad_h, grad_c=None)`` returns
    ``(dX, dH_prev, dC_prev)`` and caches the parameter gradients.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        weight_init: str = "xavier",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize a single LSTM cell."""
        _validate_sizes(input_size, hidden_size, weight_init)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_init = weight_init
        self.random_state = random_state
        gate_size = 4 * hidden_size
        self.W_ih, self.W_hh = _init_weights(
            input_size, hidden_size, gate_size, weight_init, random_state
        )
        self.b = zeros((gate_size,)) if bias else None

        self._x: FloatArray | None = None
        self._h_prev: FloatArray | None = None
        self._c_prev: FloatArray | None = None
        self._h: FloatArray | None = None
        self._c: FloatArray | None = None
        self._gates: tuple[FloatArray, ...] | None = None
        self._dW_ih: FloatArray | None = None
        self._dW_hh: FloatArray | None = None
        self._db: FloatArray | None = None

    def forward(
        self,
        x: FloatArray,
        state: tuple[FloatArray, FloatArray] | None = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Compute one LSTM step, using zero hidden/cell states by default."""
        if x.ndim != 2:
            raise ValueError(f"x must have shape (N, {self.input_size}), got {x.shape}")
        if state is None:
            h_prev = c_prev = None
        else:
            h_prev, c_prev = state
        h_prev = _initial_state(h_prev, x.shape[0], self.hidden_size, "h_prev")
        c_prev = _initial_state(c_prev, x.shape[0], self.hidden_size, "c_prev")
        _check_step_input(x, h_prev, self.input_size, self.hidden_size)

        self._x, self._h_prev, self._c_prev = x, h_prev, c_prev
        self._h, self._c, self._gates = _lstm_cell_forward(
            x, h_prev, c_prev, self.W_ih, self.W_hh, self.b
        )
        return self._h, self._c

    def backward(
        self,
        grad_h: FloatArray,
        grad_c: FloatArray | None = None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Cache parameter gradients and return ``(dX, dH_prev, dC_prev)``."""
        expected = self._h.shape
        if grad_h.shape != expected:
            raise ValueError(f"grad_h must have shape {expected}, got {grad_h.shape}")
        grad_c = _initial_state(grad_c, expected[0], expected[1], "grad_c")
        (
            dx,
            grad_h_prev,
            grad_c_prev,
            self._dW_ih,
            self._dW_hh,
            db,
        ) = _lstm_cell_backward(
            self._x,
            self._h_prev,
            self._c_prev,
            self._c,
            self._gates,
            self.W_ih,
            self.W_hh,
            grad_h,
            grad_c,
        )
        self._db = db if self.bias else None
        return dx, grad_h_prev, grad_c_prev

    def parameters(self) -> list[FloatArray]:
        """Return ``[W_ih, W_hh, b]`` when biased, otherwise both weights."""
        params = [self.W_ih, self.W_hh]
        return [*params, self.b] if self.bias else params

    def grads(self) -> list[FloatArray]:
        """Return gradients in the same order as :meth:`parameters`."""
        grads = [self._dW_ih, self._dW_hh]
        return [*grads, self._db] if self.bias else grads


class LSTM(LSTMCell):
    r"""Batch-first LSTM with a full manual BPTT pass.

    Input and output shapes are ``(N, T, input_size)`` and
    ``(N, T, hidden_size)``. Final states are :attr:`h_n`/:attr:`c_n`;
    initial-state gradients are :attr:`grad_h0`/:attr:`grad_c0`.

    ``grad_h_n`` and ``grad_c_n`` passed to :meth:`backward` represent
    additional loss edges attached directly to the final states.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        weight_init: str = "xavier",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """See :class:`LSTMCell`; this layer unrolls the cell over time."""
        super().__init__(input_size, hidden_size, bias, weight_init, random_state)
        self.h_n: FloatArray | None = None
        self.c_n: FloatArray | None = None
        self.grad_h0: FloatArray | None = None
        self.grad_c0: FloatArray | None = None
        self._hidden: FloatArray | None = None
        self._cell: FloatArray | None = None
        self._gate_sequence: list[tuple[FloatArray, ...]] | None = None

    def forward(
        self,
        x: FloatArray,
        state: tuple[FloatArray, FloatArray] | None = None,
    ) -> FloatArray:
        """Unroll the LSTM recurrence over a batch-first input sequence."""
        _check_sequence_input(x, self.input_size)
        n, timesteps, _ = x.shape
        if state is None:
            h0 = c0 = None
        else:
            h0, c0 = state
        h0 = _initial_state(h0, n, self.hidden_size, "h0")
        c0 = _initial_state(c0, n, self.hidden_size, "c0")

        hidden = np.empty((n, timesteps + 1, self.hidden_size), dtype=np.float64)
        cell = np.empty_like(hidden)
        hidden[:, 0], cell[:, 0] = h0, c0
        gate_sequence: list[tuple[FloatArray, ...]] = []
        for t in range(timesteps):
            hidden[:, t + 1], cell[:, t + 1], gates = _lstm_cell_forward(
                x[:, t],
                hidden[:, t],
                cell[:, t],
                self.W_ih,
                self.W_hh,
                self.b,
            )
            gate_sequence.append(gates)

        self._x = x
        self._hidden, self._cell = hidden, cell
        self._gate_sequence = gate_sequence
        self.h_n = hidden[:, -1].copy()
        self.c_n = cell[:, -1].copy()
        return hidden[:, 1:]

    def backward(
        self,
        grad_output: FloatArray,
        grad_h_n: FloatArray | None = None,
        grad_c_n: FloatArray | None = None,
    ) -> FloatArray:
        """Run full reverse-time BPTT, returning ``dX``."""
        n, timesteps, _ = self._x.shape
        expected = (n, timesteps, self.hidden_size)
        if grad_output.shape != expected:
            raise ValueError(
                f"grad_output must have shape {expected}, got {grad_output.shape}"
            )
        grad_h_future = _initial_state(grad_h_n, n, self.hidden_size, "grad_h_n").copy()
        grad_c_future = _initial_state(grad_c_n, n, self.hidden_size, "grad_c_n").copy()
        dx = np.empty_like(self._x, dtype=np.float64)
        self._dW_ih = np.zeros_like(self.W_ih)
        self._dW_hh = np.zeros_like(self.W_hh)
        db = np.zeros(4 * self.hidden_size, dtype=np.float64)

        for t in range(timesteps - 1, -1, -1):
            grad_h = grad_output[:, t] + grad_h_future
            (
                dx[:, t],
                grad_h_future,
                grad_c_future,
                dW_ih_t,
                dW_hh_t,
                db_t,
            ) = _lstm_cell_backward(
                self._x[:, t],
                self._hidden[:, t],
                self._cell[:, t],
                self._cell[:, t + 1],
                self._gate_sequence[t],
                self.W_ih,
                self.W_hh,
                grad_h,
                grad_c_future,
            )
            self._dW_ih += dW_ih_t
            self._dW_hh += dW_hh_t
            db += db_t

        self.grad_h0, self.grad_c0 = grad_h_future, grad_c_future
        self._db = db if self.bias else None
        return dx
