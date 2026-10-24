"""Tests for the NumPy AdamW optimizer in source/training/optimizer.py."""

import numpy as np
import pytest

from training.optimizer import AdamW


def test_initializes_independent_moments():
    w1 = np.ones((2, 3), dtype=np.float64)
    w2 = np.ones(4, dtype=np.float64)
    g1 = np.zeros_like(w1)
    g2 = np.zeros_like(w2)

    optimizer = AdamW([(w1, g1, True), (w2, g2, False)])

    assert optimizer.t == 0
    assert len(optimizer.m) == len(optimizer.v) == 2
    assert optimizer.m[0].shape == optimizer.v[0].shape == (2, 3)
    assert optimizer.m[1].shape == optimizer.v[1].shape == (4,)
    assert all(np.all(moment == 0) for moment in optimizer.m + optimizer.v)


def test_first_step_without_weight_decay():
    w = np.array([2.0, -3.0, 5.0], dtype=np.float64)
    dw = np.array([0.2, -0.4, 0.0], dtype=np.float64)
    original = w.copy()
    lr, eps = 0.01, 1e-8
    optimizer = AdamW([(w, dw, False)], lr=lr, eps=eps)

    optimizer.step()

    # On the first step, bias-corrected m = dw and v = dw**2.
    expected = original - lr * dw / (np.abs(dw) + eps)
    np.testing.assert_allclose(w, expected, rtol=1e-12, atol=1e-12)
    assert optimizer.t == 1


def test_weight_decay_can_be_disabled_per_parameter():
    weights = np.array([2.0, -4.0], dtype=np.float64)
    bias = np.array([2.0, -4.0], dtype=np.float64)
    gw = np.zeros_like(weights)
    gb = np.zeros_like(bias)
    optimizer = AdamW(
        [(weights, gw, True), (bias, gb, False)],
        lr=0.1,
        weight_decay=0.2,
    )

    optimizer.step()

    np.testing.assert_allclose(weights, np.array([1.96, -3.92]), atol=1e-12)
    np.testing.assert_array_equal(bias, np.array([2.0, -4.0]))


def test_decay_uses_pre_update_weights():
    # Checks the exact, simultaneous AdamW update (not decay applied afterward).
    w = np.array([2.0], dtype=np.float64)
    dw = np.array([0.5], dtype=np.float64)
    optimizer = AdamW([(w, dw, True)], lr=0.1, eps=1e-8, weight_decay=0.2)

    optimizer.step()

    expected = 2.0 - 0.1 * (0.5 / (0.5 + 1e-8) + 0.2 * 2.0)
    np.testing.assert_allclose(w, [expected], rtol=1e-12, atol=1e-12)


def test_two_steps_update_moments_and_bias_correction():
    w = np.array([1.0, -2.0], dtype=np.float64)
    dw = np.array([0.3, -0.1], dtype=np.float64)
    optimizer = AdamW([(w, dw, False)], lr=0.01)

    optimizer.step()
    np.testing.assert_allclose(optimizer.m[0], 0.1 * dw, atol=1e-15)
    np.testing.assert_allclose(optimizer.v[0], 0.001 * dw**2, atol=1e-15)

    # Change the gradient for the second training step.
    dw[:] = [-0.2, 0.4]
    optimizer.step()

    m_expected = 0.9 * (0.1 * np.array([0.3, -0.1])) + 0.1 * dw
    v_expected = 0.999 * (0.001 * np.array([0.3, -0.1]) ** 2) + 0.001 * dw**2
    assert optimizer.t == 2
    np.testing.assert_allclose(optimizer.m[0], m_expected, atol=1e-15)
    np.testing.assert_allclose(optimizer.v[0], v_expected, atol=1e-15)


def test_moments_are_independent_between_parameters():
    w1 = np.array([1.0, 1.0], dtype=np.float64)
    w2 = np.array([1.0, 1.0], dtype=np.float64)
    g1 = np.array([0.5, 0.0], dtype=np.float64)
    g2 = np.array([0.0, -0.25], dtype=np.float64)
    optimizer = AdamW([(w1, g1, False), (w2, g2, False)])

    optimizer.step()

    np.testing.assert_allclose(optimizer.m[0], 0.1 * g1, atol=1e-15)
    np.testing.assert_allclose(optimizer.m[1], 0.1 * g2, atol=1e-15)
    np.testing.assert_allclose(optimizer.v[0], 0.001 * g1**2, atol=1e-15)
    np.testing.assert_allclose(optimizer.v[1], 0.001 * g2**2, atol=1e-15)
    assert optimizer.t == 1  # Not one step per parameter.


def test_step_updates_parameters_in_place():
    w = np.array([1.0, -1.0], dtype=np.float64)
    dw = np.array([0.2, -0.1], dtype=np.float64)
    alias = w
    optimizer = AdamW([(w, dw, False)])

    result = optimizer.step()

    assert result is None
    assert alias is w
    assert not np.array_equal(alias, np.array([1.0, -1.0]))


def test_step_does_not_mutate_gradients():
    w = np.array([1.0, 2.0], dtype=np.float64)
    dw = np.array([-0.2, 0.3], dtype=np.float64)
    original = dw.copy()
    optimizer = AdamW([(w, dw, True)])

    optimizer.step()

    np.testing.assert_array_equal(dw, original)


def test_multiple_parameter_shapes():
    rng = np.random.default_rng(12)
    embedding = rng.normal(size=(10, 3))
    linear = rng.normal(size=(3, 10))
    bias = rng.normal(size=10)
    parameters = [
        (embedding, np.ones_like(embedding), True),
        (linear, np.ones_like(linear), True),
        (bias, np.ones_like(bias), False),
    ]
    originals = [w.copy() for w, _, _ in parameters]
    optimizer = AdamW(parameters)

    optimizer.step()

    assert [w.shape for w, _, _ in parameters] == [(10, 3), (3, 10), (10,)]
    for (w, _, _), original in zip(parameters, originals):
        assert np.all(np.isfinite(w))
        assert not np.array_equal(w, original)


def test_reuses_live_gradient_array_on_next_step():
    w = np.array([1.0], dtype=np.float64)
    dw = np.array([0.2], dtype=np.float64)
    optimizer = AdamW([(w, dw, False)], lr=0.01)

    optimizer.step()
    dw.fill(0.0)  # Like zero_grad() on the associated layer.
    optimizer.step()

    np.testing.assert_allclose(optimizer.m[0], [0.018], atol=1e-15)
    assert optimizer.t == 2


def test_parameters_generator_supported():
    w = np.array([1.0], dtype=np.float64)
    dw = np.array([0.2], dtype=np.float64)
    params = ((weight, grad, decay) for weight, grad, decay in [(w, dw, False)])
    optimizer = AdamW(params)

    optimizer.step()

    assert optimizer.t == 1
    assert len(optimizer.m) == 1
    assert w[0] < 1.0

# python -m pytest tests/test_optimizer.py -v