"""Tests for nn.rmsnorm.RMSNorm.

Run from the project root:
    python -m pytest tests/test_rmsnorm.py -v

Expected interface:
    RMSNorm(embedding_dim: int, eps: float = 1e-6)
    forward(x), backward(grad_output), zero_grad(), parameters()
    gamma, grad_gamma
"""

import numpy as np
import pytest

from nn.rmsnorm import RMSNorm


def objective(layer, x, upstream):
    return float(np.sum(layer.forward(x) * upstream))


def numerical_input_gradient(layer, x, upstream, eps=1e-5):
    result = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        xp = x.copy()
        xm = x.copy()
        xp[index] += eps
        xm[index] -= eps
        result[index] = (objective(layer, xp, upstream) - objective(layer, xm, upstream)) / (2 * eps)
    return result


@pytest.mark.parametrize("shape", [(1, 1, 4), (2, 5, 8), (3, 4), (4,)])
def test_forward_shape_and_finiteness(shape):
    rng = np.random.default_rng(10)
    x = rng.normal(size=shape).astype(np.float32)
    layer = RMSNorm(shape[-1])
    y = layer.forward(x)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_known_forward_values():
    layer = RMSNorm(4)
    x = np.array([[[2.0, 4.0, 6.0, 8.0]]], dtype=np.float32)
    expected = x / np.sqrt(30.0 + layer.eps)
    np.testing.assert_allclose(layer.forward(x), expected, rtol=1e-6, atol=1e-6)


def test_normalization_has_unit_mean_square_with_negligible_epsilon():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2, 3, 8)).astype(np.float64)
    layer = RMSNorm(8, eps=1e-12)
    y = layer.forward(x)
    np.testing.assert_allclose(np.mean(y**2, axis=-1), 1.0, atol=1e-9)


def test_gamma_scales_each_feature():
    layer = RMSNorm(4)
    layer.gamma[:] = [2.0, 0.5, -1.0, 0.0]
    x = np.array([[[1.0, -2.0, 3.0, 4.0]]], dtype=np.float32)
    inv_rms = 1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + layer.eps)
    expected = x * inv_rms * layer.gamma
    np.testing.assert_allclose(layer.forward(x), expected, rtol=1e-6)


def test_zero_input_is_finite_and_zero():
    layer = RMSNorm(6)
    x = np.zeros((2, 3, 6), dtype=np.float32)
    y = layer.forward(x)
    np.testing.assert_array_equal(y, np.zeros_like(y))
    grad = layer.backward(np.ones_like(y))
    assert np.all(np.isfinite(grad))
    np.testing.assert_array_equal(layer.grad_gamma, np.zeros_like(layer.grad_gamma))


def test_wrong_feature_dimension_raises():
    with pytest.raises(ValueError):
        RMSNorm(4).forward(np.ones((2, 3, 5), dtype=np.float32))


@pytest.mark.parametrize("dim", [0, -1, -8])
def test_invalid_embedding_dim(dim):
    with pytest.raises(ValueError):
        RMSNorm(dim)


@pytest.mark.parametrize("eps", [0.0, -1e-6, np.inf, np.nan])
def test_invalid_epsilon(eps):
    with pytest.raises(ValueError):
        RMSNorm(4, eps=eps)


@pytest.mark.parametrize("shape", [(2, 3, 4), (3, 6), (4,)])
def test_numerical_input_gradient(shape):
    rng = np.random.default_rng(22)
    x = rng.normal(size=shape).astype(np.float64)
    upstream = rng.normal(size=shape).astype(np.float64)
    layer = RMSNorm(shape[-1])
    layer.gamma[:] = rng.uniform(0.5, 1.5, size=shape[-1])

    layer.forward(x)
    analytical = layer.backward(upstream).copy()
    numerical = numerical_input_gradient(layer, x, upstream)
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-6)


def test_numerical_gamma_gradient():
    rng = np.random.default_rng(33)
    x = rng.normal(size=(2, 3, 4)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    layer = RMSNorm(4)
    layer.gamma[:] = [0.8, 1.2, 0.4, 1.5]

    layer.forward(x)
    layer.backward(upstream)
    analytical = layer.grad_gamma.copy()

    # gamma is float32, so use a moderately sized finite-difference step.
    eps = 1e-3
    numerical = np.zeros_like(layer.gamma, dtype=np.float64)
    for i in range(layer.gamma.size):
        original = float(layer.gamma[i])
        layer.gamma[i] = original + eps
        plus = objective(layer, x, upstream)
        layer.gamma[i] = original - eps
        minus = objective(layer, x, upstream)
        layer.gamma[i] = original
        numerical[i] = (plus - minus) / (2 * eps)

    np.testing.assert_allclose(analytical, numerical, rtol=5e-4, atol=2e-4)


def test_backward_shapes_and_finiteness():
    rng = np.random.default_rng(44)
    x = rng.normal(size=(2, 5, 6)).astype(np.float32)
    layer = RMSNorm(6)
    layer.forward(x)
    dx = layer.backward(rng.normal(size=x.shape).astype(np.float32))
    assert dx.shape == x.shape
    assert layer.grad_gamma.shape == (6,)
    assert np.all(np.isfinite(dx))
    assert np.all(np.isfinite(layer.grad_gamma))


def test_backward_accumulates_gamma_gradient():
    rng = np.random.default_rng(55)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer = RMSNorm(4)
    layer.forward(x)
    layer.backward(upstream)
    first = layer.grad_gamma.copy()
    layer.backward(upstream)
    np.testing.assert_allclose(layer.grad_gamma, 2 * first, rtol=1e-6, atol=1e-6)


def test_zero_grad_resets_gamma_gradient():
    rng = np.random.default_rng(66)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    layer = RMSNorm(4)
    layer.forward(x)
    layer.backward(np.ones_like(x))
    assert np.any(layer.grad_gamma != 0)
    layer.zero_grad()
    np.testing.assert_array_equal(layer.grad_gamma, np.zeros_like(layer.gamma))


def test_parameters_expose_gamma_without_weight_decay():
    layer = RMSNorm(4)
    params = layer.parameters()
    assert len(params) == 1
    weights, gradients, decay = params[0]
    assert weights is layer.gamma
    assert gradients is layer.grad_gamma
    assert decay is False
# python -m pytest tests/test_rmsnorm.py -v