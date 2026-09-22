"""Tests for recurrent cells and full backpropagation through time."""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import LSTM, RNN, LSTMCell, RNNCell
from tests.helpers.gradcheck import gradient_check


class TestContract:
    @pytest.mark.parametrize("cls", [RNNCell, RNN, LSTMCell, LSTM])
    def test_invalid_sizes_and_initialization_raise(self, cls: type) -> None:
        with pytest.raises(ValueError, match="input_size"):
            cls(0, 2)
        with pytest.raises(ValueError, match="hidden_size"):
            cls(2, 0)
        with pytest.raises(ValueError, match="weight_init"):
            cls(2, 3, weight_init="he")

    @pytest.mark.parametrize("cls", [RNNCell, RNN, LSTMCell, LSTM])
    def test_parameters_and_grads_have_matching_shapes(self, cls: type) -> None:
        layer = cls(2, 3, random_state=0)
        if cls is RNNCell:
            y = layer.forward(np.ones((4, 2)))
            layer.backward(np.ones_like(y))
        elif cls is LSTMCell:
            h, c = layer.forward(np.ones((4, 2)))
            layer.backward(np.ones_like(h), np.ones_like(c))
        else:
            y = layer.forward(np.ones((4, 5, 2)))
            layer.backward(np.ones_like(y))

        assert len(layer.parameters()) == len(layer.grads()) == 3
        for parameter, grad in zip(layer.parameters(), layer.grads(), strict=True):
            assert parameter.shape == grad.shape

    @pytest.mark.parametrize("cls", [RNNCell, RNN, LSTMCell, LSTM])
    def test_bias_false_exposes_only_weight_matrices(self, cls: type) -> None:
        layer = cls(2, 3, bias=False, random_state=0)
        assert layer.b is None
        assert len(layer.parameters()) == 2

    @pytest.mark.parametrize("cls", [RNN, LSTM])
    def test_sequence_input_must_be_batch_first_and_non_empty(self, cls: type) -> None:
        layer = cls(2, 3, random_state=0)
        with pytest.raises(ValueError, match="shape"):
            layer.forward(np.ones((4, 2)))
        with pytest.raises(ValueError, match="timestep"):
            layer.forward(np.ones((4, 0, 2)))

    def test_wrong_initial_state_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="h0"):
            RNN(2, 3).forward(np.ones((4, 5, 2)), np.zeros((1, 3)))
        with pytest.raises(ValueError, match="c0"):
            LSTM(2, 3).forward(np.ones((4, 5, 2)), (np.zeros((4, 3)), np.zeros((1, 3))))

    def test_sequence_backward_shape_is_validated(self) -> None:
        layer = RNN(2, 3, random_state=0)
        layer.forward(np.ones((4, 5, 2)))
        with pytest.raises(ValueError, match="grad_output"):
            layer.backward(np.ones((4, 3)))


class TestAnalytic:
    def test_rnn_cell_matches_direct_equation(self) -> None:
        cell = RNNCell(1, 1, weight_init="zeros")
        cell.W_ih[:] = 2.0
        cell.W_hh[:] = 3.0
        cell.b[:] = 0.5
        result = cell.forward(np.array([[1.0]]), np.array([[0.1]]))
        np.testing.assert_allclose(result, np.tanh([[2.8]]))

    def test_rnn_bptt_accumulates_future_and_parameter_gradients(self) -> None:
        layer = RNN(1, 1, weight_init="zeros")
        layer.W_hh[:] = 1.0
        output = layer.forward(np.ones((1, 3, 1)))
        dx = layer.backward(np.ones_like(output))

        np.testing.assert_array_equal(output, np.zeros((1, 3, 1)))
        np.testing.assert_array_equal(dx, np.zeros((1, 3, 1)))
        np.testing.assert_allclose(layer.grad_h0, [[3.0]])
        np.testing.assert_allclose(layer.grads()[0], [[6.0]])
        np.testing.assert_allclose(layer.grads()[1], [[0.0]])
        np.testing.assert_allclose(layer.grads()[2], [6.0])

    def test_lstm_cell_zero_preactivations_have_half_open_gates(self) -> None:
        cell = LSTMCell(1, 1, weight_init="zeros")
        h, c = cell.forward(np.array([[7.0]]), (np.array([[0.0]]), np.array([[2.0]])))
        np.testing.assert_allclose(c, [[1.0]])
        np.testing.assert_allclose(h, 0.5 * np.tanh([[1.0]]))

    def test_sequence_layers_return_all_and_record_final_states(self) -> None:
        x = np.ones((2, 4, 3))
        rnn = RNN(3, 5, random_state=0)
        lstm = LSTM(3, 5, random_state=0)

        rnn_output = rnn.forward(x)
        lstm_output = lstm.forward(x)

        assert rnn_output.shape == lstm_output.shape == (2, 4, 5)
        np.testing.assert_array_equal(rnn.h_n, rnn_output[:, -1])
        np.testing.assert_array_equal(lstm.h_n, lstm_output[:, -1])
        assert lstm.c_n.shape == (2, 5)

    def test_terminal_gradients_are_additional_to_sequence_gradient(self) -> None:
        x = np.zeros((1, 2, 1))
        layer = RNN(1, 1, weight_init="zeros")
        layer.W_hh[:] = 1.0
        output = layer.forward(x)
        layer.backward(np.ones_like(output), grad_h_n=np.ones((1, 1)))
        np.testing.assert_allclose(layer.grad_h0, [[3.0]])


class TestRNNGradient:
    def setup_method(self) -> None:
        rng = np.random.default_rng(1)
        self.x = rng.standard_normal((2, 3, 2))
        self.h0 = rng.standard_normal((2, 2))
        self.direction = rng.standard_normal((2, 3, 2))
        self.final_direction = rng.standard_normal((2, 2))
        self.layer = RNN(2, 2, random_state=2)
        self.layer.forward(self.x, self.h0)
        self.dx = self.layer.backward(self.direction, self.final_direction).copy()
        self.dW_ih, self.dW_hh, self.db = [g.copy() for g in self.layer.grads()]
        self.dh0 = self.layer.grad_h0.copy()

    def objective(self, x: np.ndarray, h0: np.ndarray) -> float:
        output = self.layer.forward(x, h0)
        return float(
            np.sum(output * self.direction)
            + np.sum(self.layer.h_n * self.final_direction)
        )

    def parameter_objective(self, name: str, value: np.ndarray) -> float:
        setattr(self.layer, name, value)
        return self.objective(self.x, self.h0)

    def test_dX(self) -> None:
        gradient_check(lambda value: self.objective(value, self.h0), self.dx, self.x)

    def test_dH0(self) -> None:
        gradient_check(lambda value: self.objective(self.x, value), self.dh0, self.h0)

    def test_dW_ih(self) -> None:
        gradient_check(
            lambda value: self.parameter_objective("W_ih", value),
            self.dW_ih,
            self.layer.W_ih,
        )

    def test_dW_hh(self) -> None:
        gradient_check(
            lambda value: self.parameter_objective("W_hh", value),
            self.dW_hh,
            self.layer.W_hh,
        )

    def test_db(self) -> None:
        gradient_check(
            lambda value: self.parameter_objective("b", value), self.db, self.layer.b
        )


class TestLSTMGradient:
    def setup_method(self) -> None:
        rng = np.random.default_rng(3)
        self.x = rng.standard_normal((2, 2, 2))
        self.h0 = rng.standard_normal((2, 2))
        self.c0 = rng.standard_normal((2, 2))
        self.direction = rng.standard_normal((2, 2, 2))
        self.final_h_direction = rng.standard_normal((2, 2))
        self.final_c_direction = rng.standard_normal((2, 2))
        self.layer = LSTM(2, 2, random_state=4)
        self.layer.forward(self.x, (self.h0, self.c0))
        self.dx = self.layer.backward(
            self.direction, self.final_h_direction, self.final_c_direction
        ).copy()
        self.dW_ih, self.dW_hh, self.db = [g.copy() for g in self.layer.grads()]
        self.dh0 = self.layer.grad_h0.copy()
        self.dc0 = self.layer.grad_c0.copy()

    def objective(self, x: np.ndarray, h0: np.ndarray, c0: np.ndarray) -> float:
        output = self.layer.forward(x, (h0, c0))
        return float(
            np.sum(output * self.direction)
            + np.sum(self.layer.h_n * self.final_h_direction)
            + np.sum(self.layer.c_n * self.final_c_direction)
        )

    def parameter_objective(self, name: str, value: np.ndarray) -> float:
        setattr(self.layer, name, value)
        return self.objective(self.x, self.h0, self.c0)

    def test_dX(self) -> None:
        gradient_check(
            lambda value: self.objective(value, self.h0, self.c0), self.dx, self.x
        )

    def test_dH0(self) -> None:
        gradient_check(
            lambda value: self.objective(self.x, value, self.c0), self.dh0, self.h0
        )

    def test_dC0(self) -> None:
        gradient_check(
            lambda value: self.objective(self.x, self.h0, value), self.dc0, self.c0
        )

    def test_dW_ih(self) -> None:
        gradient_check(
            lambda value: self.parameter_objective("W_ih", value),
            self.dW_ih,
            self.layer.W_ih,
        )

    def test_dW_hh(self) -> None:
        gradient_check(
            lambda value: self.parameter_objective("W_hh", value),
            self.dW_hh,
            self.layer.W_hh,
        )

    def test_db(self) -> None:
        gradient_check(
            lambda value: self.parameter_objective("b", value),
            self.db,
            self.layer.b,
        )


class TestDeterminism:
    @pytest.mark.parametrize("cls", [RNNCell, RNN, LSTMCell, LSTM])
    def test_same_seed_gives_identical_parameters(self, cls: type) -> None:
        first = cls(3, 4, random_state=0)
        second = cls(3, 4, random_state=0)
        for a, b in zip(first.parameters(), second.parameters(), strict=True):
            np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize("cls", [RNNCell, RNN, LSTMCell, LSTM])
    def test_zeros_initialization(self, cls: type) -> None:
        layer = cls(3, 4, weight_init="zeros")
        for parameter in layer.parameters():
            np.testing.assert_array_equal(parameter, np.zeros_like(parameter))
