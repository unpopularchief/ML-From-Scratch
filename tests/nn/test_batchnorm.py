"""Tests for scratchgrad.nn.layers.BatchNorm1d.

Tiers (plan.md section 3, docs/derivations/dropout_batchnorm.md section 4):
a hand-computed forward pass, separate gradient checks for the train-mode
and eval-mode backward formulas, the running-stats Bessel-correction
update, the parameters()/grads() contract, and invalid-hyperparameter
errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from scratchgrad.nn import BatchNorm1d
from scratchgrad.nn.layers.batchnorm import (
    _batchnorm_backward_eval,
    _batchnorm_backward_train,
    _batchnorm_forward_eval,
    _batchnorm_forward_train,
)
from tests.helpers.gradcheck import gradient_check


class TestContract:
    def test_non_positive_num_features_raises(self) -> None:
        with pytest.raises(ValueError, match="num_features"):
            BatchNorm1d(0)
        with pytest.raises(ValueError, match="num_features"):
            BatchNorm1d(-1)

    def test_training_mode_batch_size_one_raises(self) -> None:
        bn = BatchNorm1d(3)
        with pytest.raises(ValueError, match="batch size"):
            bn.forward(np.ones((1, 3)))

    def test_eval_mode_batch_size_one_does_not_raise(self) -> None:
        bn = BatchNorm1d(3).eval()
        bn.forward(np.ones((1, 3)))  # should not raise

    def test_eval_mode_forward_and_backward_do_not_touch_running_stats(
        self, rng: np.random.Generator
    ) -> None:
        bn = BatchNorm1d(4).eval()
        running_mean_before = bn.running_mean.copy()
        running_var_before = bn.running_var.copy()

        bn.forward(rng.standard_normal((5, 4)))
        dx = bn.backward(np.ones((5, 4)))

        assert dx.shape == (5, 4)
        np.testing.assert_array_equal(bn.running_mean, running_mean_before)
        np.testing.assert_array_equal(bn.running_var, running_var_before)

    def test_gamma_starts_at_ones_beta_at_zeros(self) -> None:
        bn = BatchNorm1d(4)
        np.testing.assert_array_equal(bn.gamma, np.ones(4))
        np.testing.assert_array_equal(bn.beta, np.zeros(4))

    def test_running_stats_start_at_zero_mean_unit_var(self) -> None:
        bn = BatchNorm1d(4)
        np.testing.assert_array_equal(bn.running_mean, np.zeros(4))
        np.testing.assert_array_equal(bn.running_var, np.ones(4))

    def test_defaults_to_training_mode(self) -> None:
        assert BatchNorm1d(3).training is True

    def test_parameters_and_grads_same_order_and_shape(
        self, rng: np.random.Generator
    ) -> None:
        bn = BatchNorm1d(4)
        x = rng.standard_normal((5, 4))
        bn.forward(x)
        bn.backward(np.ones((5, 4)))

        params, grads = bn.parameters(), bn.grads()
        assert len(params) == len(grads) == 2
        for p, g in zip(params, grads, strict=True):
            assert p.shape == g.shape

    def test_forward_backward_do_not_mutate_hyperparameters(
        self, rng: np.random.Generator
    ) -> None:
        bn = BatchNorm1d(4, eps=1e-4, momentum=0.2)
        bn.forward(rng.standard_normal((5, 4)))
        bn.backward(np.ones((5, 4)))
        assert bn.num_features == 4
        assert bn.eps == 1e-4
        assert bn.momentum == 0.2


class TestAnalytic:
    def test_train_forward_gives_zero_mean_unit_var_before_affine(self) -> None:
        X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        bn = BatchNorm1d(3)  # gamma=1, beta=0 -- output equals x_hat directly

        y = bn.forward(X)

        np.testing.assert_allclose(y.mean(axis=0), np.zeros(3), atol=1e-6)
        np.testing.assert_allclose(np.mean(y**2, axis=0), np.ones(3), atol=1e-4)

    def test_train_forward_matches_hand_computation(self) -> None:
        X = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
        gamma, beta, eps = np.ones(3), np.zeros(3), 0.0
        y, x_hat, std, var = _batchnorm_forward_train(X, gamma, beta, eps)

        np.testing.assert_allclose(var, np.ones(3))  # mean((+-1)**2) = 1
        np.testing.assert_allclose(std, np.ones(3))
        np.testing.assert_allclose(x_hat, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(y, x_hat)

    def test_running_stats_update_uses_bessel_corrected_var(self) -> None:
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # n=3
        bn = BatchNorm1d(2, momentum=1.0)  # momentum=1 -> running := batch stat exactly
        bn.forward(X)

        expected_mean = X.mean(axis=0)
        biased_var = X.var(axis=0)
        unbiased_var = biased_var * 3 / 2  # n/(n-1)

        np.testing.assert_allclose(bn.running_mean, expected_mean)
        np.testing.assert_allclose(bn.running_var, unbiased_var)


class TestGradient:
    def test_train_mode_gradients_match_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.standard_normal((6, 4))
        gamma = rng.standard_normal(4)
        beta = rng.standard_normal(4)
        eps = 1e-5
        R = rng.standard_normal((6, 4))

        _, x_hat, std, _ = _batchnorm_forward_train(X, gamma, beta, eps)
        dX, dgamma, dbeta = _batchnorm_backward_train(x_hat, std, gamma, R)

        def f_X(X_: np.ndarray) -> float:
            y, _, _, _ = _batchnorm_forward_train(X_, gamma, beta, eps)
            return float(np.sum(y * R))

        def f_gamma(gamma_: np.ndarray) -> float:
            y, _, _, _ = _batchnorm_forward_train(X, gamma_, beta, eps)
            return float(np.sum(y * R))

        def f_beta(beta_: np.ndarray) -> float:
            y, _, _, _ = _batchnorm_forward_train(X, gamma, beta_, eps)
            return float(np.sum(y * R))

        gradient_check(f_X, dX, X)
        gradient_check(f_gamma, dgamma, gamma)
        gradient_check(f_beta, dbeta, beta)

    def test_eval_mode_gradients_match_finite_differences(
        self, rng: np.random.Generator
    ) -> None:
        X = rng.standard_normal((6, 4))
        gamma = rng.standard_normal(4)
        beta = rng.standard_normal(4)
        running_mean = rng.standard_normal(4)
        running_var = np.abs(rng.standard_normal(4)) + 0.5
        eps = 1e-5
        R = rng.standard_normal((6, 4))

        _, x_hat, std = _batchnorm_forward_eval(
            X, gamma, beta, running_mean, running_var, eps
        )
        dX, dgamma, dbeta = _batchnorm_backward_eval(x_hat, std, gamma, R)

        def f_X(X_: np.ndarray) -> float:
            y, _, _ = _batchnorm_forward_eval(
                X_, gamma, beta, running_mean, running_var, eps
            )
            return float(np.sum(y * R))

        def f_gamma(gamma_: np.ndarray) -> float:
            y, _, _ = _batchnorm_forward_eval(
                X, gamma_, beta, running_mean, running_var, eps
            )
            return float(np.sum(y * R))

        def f_beta(beta_: np.ndarray) -> float:
            y, _, _ = _batchnorm_forward_eval(
                X, gamma, beta_, running_mean, running_var, eps
            )
            return float(np.sum(y * R))

        gradient_check(f_X, dX, X)
        gradient_check(f_gamma, dgamma, gamma)
        gradient_check(f_beta, dbeta, beta)


class TestTrainEvalDispatch:
    def test_module_backward_dispatches_on_forwards_own_mode(
        self, rng: np.random.Generator
    ) -> None:
        # backward must use whichever mode `forward` actually ran in, even
        # if train()/eval() is toggled in between -- see the derivation
        # doc section 3's note on caching `_was_training`.
        X = rng.standard_normal((6, 4))
        grad_output = rng.standard_normal((6, 4))

        bn_train = BatchNorm1d(4)
        bn_train.forward(X)
        bn_train.eval()  # toggled after forward, before backward
        dx_toggled = bn_train.backward(grad_output)

        bn_reference = BatchNorm1d(4)
        bn_reference.forward(X)
        dx_reference = bn_reference.backward(grad_output)  # still training throughout

        np.testing.assert_array_equal(dx_toggled, dx_reference)
