"""Tests for NumPy SwiGLU feed-forward layer.

Run from project root:
    python -m pytest tests/test_mlp.py -v

Expected: nn.mlp.SwiGLU with forward, backward, zero_grad, parameters,
and gate/up/down Linear layers.
"""

import numpy as np
import pytest

from nn.mlp import SwiGLU


def make_mlp(embedding_dim=4, hidden_dim=8, seed=42):
    return SwiGLU(embedding_dim, hidden_dim, np.random.default_rng(seed))


def objective(layer, x, upstream):
    return float(np.sum(layer.forward(x) * upstream))


@pytest.mark.parametrize(
    "shape,hidden_dim",
    [((1, 1, 4), 8), ((2, 3, 4), 12), ((3, 5, 6), 10)],
)
def test_forward_shape_and_finite(shape, hidden_dim):
    layer = make_mlp(shape[-1], hidden_dim)
    x = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    output = layer.forward(x)
    assert output.shape == x.shape
    assert np.all(np.isfinite(output))
    assert layer.gate_vals.shape == (*shape[:-1], hidden_dim)
    assert layer.up_vals.shape == (*shape[:-1], hidden_dim)


def test_forward_matches_manual_formula():
    layer = make_mlp(seed=1)
    x = np.random.default_rng(2).normal(size=(2, 3, 4)).astype(np.float32)
    output = layer.forward(x)
    gate = x @ layer.gate.weights + layer.gate.bias
    up = x @ layer.up.weights + layer.up.bias
    sigmoid = 1.0 / (1.0 + np.exp(-gate))
    expected = ((gate * sigmoid) * up) @ layer.down.weights + layer.down.bias
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_tokens_are_independent():
    layer = make_mlp(seed=3)
    x = np.random.default_rng(4).normal(size=(2, 4, 4)).astype(np.float32)
    baseline = layer.forward(x).copy()
    modified = x.copy()
    modified[0, 3, :] += 100
    actual = layer.forward(modified)
    np.testing.assert_allclose(actual[0, :3], baseline[0, :3], atol=1e-6)
    np.testing.assert_allclose(actual[1], baseline[1], atol=1e-6)


def test_backward_shape_and_finite():
    layer = make_mlp(6, 10, seed=5)
    rng = np.random.default_rng(6)
    x = rng.normal(size=(2, 3, 6)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    grad_input = layer.backward(upstream)
    assert grad_input.shape == x.shape
    assert np.all(np.isfinite(grad_input))
    for proj in (layer.gate, layer.up, layer.down):
        assert proj.grad_weights.shape == proj.weights.shape
        assert proj.grad_bias.shape == proj.bias.shape
        assert np.all(np.isfinite(proj.grad_weights))
        assert np.all(np.isfinite(proj.grad_bias))


@pytest.mark.parametrize("shape", [(1, 2, 3), (2, 3, 4)])
def test_numerical_input_gradient(shape):
    """Central differences of sum(forward(x) * upstream) w.r.t. x."""
    layer = make_mlp(shape[-1], 5, seed=7)
    rng = np.random.default_rng(8)
    x = rng.normal(size=shape).astype(np.float64)
    upstream = rng.normal(size=shape).astype(np.float64)
    layer.forward(x)
    analytic = layer.backward(upstream).copy()
    eps = 1e-5
    numerical = np.empty_like(x)
    for ix in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[ix] += eps
        minus[ix] -= eps
        numerical[ix] = (objective(layer, plus, upstream) - objective(layer, minus, upstream)) / (2 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("projection_name", ["gate", "up", "down"])
def test_numerical_weight_gradient(projection_name):
    """Check gradients of all three independently trained weight matrices."""
    layer = make_mlp(3, 4, seed=9)
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 3, 3)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    layer.backward(upstream)
    proj = getattr(layer, projection_name)
    analytic = proj.grad_weights.copy()
    eps = 1e-3  # trainable weights use float32
    numerical = np.empty_like(analytic)
    for ix in np.ndindex(proj.weights.shape):
        original = float(proj.weights[ix])
        proj.weights[ix] = original + eps
        plus = objective(layer, x, upstream)
        proj.weights[ix] = original - eps
        minus = objective(layer, x, upstream)
        proj.weights[ix] = original
        numerical[ix] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("projection_name", ["gate", "up", "down"])
def test_numerical_bias_gradient(projection_name):
    layer = make_mlp(3, 4, seed=11)
    rng = np.random.default_rng(12)
    x = rng.normal(size=(2, 2, 3)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    layer.backward(upstream)
    proj = getattr(layer, projection_name)
    analytic = proj.grad_bias.copy()
    eps = 1e-3
    numerical = np.empty_like(analytic)
    for ix in np.ndindex(proj.bias.shape):
        original = float(proj.bias[ix])
        proj.bias[ix] = original + eps
        plus = objective(layer, x, upstream)
        proj.bias[ix] = original - eps
        minus = objective(layer, x, upstream)
        proj.bias[ix] = original
        numerical[ix] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=5e-3, atol=5e-4)


def test_gradients_accumulate():
    layer = make_mlp(seed=13)
    rng = np.random.default_rng(14)
    x = rng.normal(size=(2, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    layer.backward(upstream)
    first = [(p.grad_weights.copy(), p.grad_bias.copy()) for p in (layer.gate, layer.up, layer.down)]
    layer.backward(upstream)
    for p, (dw, db) in zip((layer.gate, layer.up, layer.down), first):
        np.testing.assert_allclose(p.grad_weights, 2 * dw, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(p.grad_bias, 2 * db, rtol=1e-6, atol=1e-6)


def test_zero_grad_clears_all_projection_gradients():
    layer = make_mlp(seed=15)
    rng = np.random.default_rng(16)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    layer.backward(upstream)
    layer.zero_grad()
    for p in (layer.gate, layer.up, layer.down):
        assert np.all(p.grad_weights == 0)
        assert np.all(p.grad_bias == 0)


def test_parameters_expose_six_arrays_and_decay_flags():
    layer = make_mlp()
    params = layer.parameters()
    expected = []
    for p in (layer.gate, layer.up, layer.down):
        expected.extend([(p.weights, p.grad_weights, True), (p.bias, p.grad_bias, False)])
    assert len(params) == 6
    for (actual_w, actual_dw, actual_decay), (w, dw, decay) in zip(params, expected):
        assert actual_w is w
        assert actual_dw is dw
        assert actual_decay is decay
#python -m pytest tests/test_mlp.py -v