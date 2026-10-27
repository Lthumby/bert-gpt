"""Tests for vectorized multi-head causal self-attention with RoPE.

Run from the project root:
    python -m pytest tests/test_multi_head_attention.py -v

Expected: nn.multi_head_attention.MultiHeadAttention supporting forward,
backward, zero_grad, parameters and caching Q, K, V, mask, attention_weights.
"""

import numpy as np
import pytest

from nn.multi_head_attention import MultiHeadAttention


def make_layer(embedding_dim=8, num_heads=2, seed=42):
    return MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        rng=np.random.default_rng(seed),
    )


def objective(layer, x, upstream):
    return float(np.sum(layer.forward(x) * upstream))


def numerical_input_gradient(layer, x, upstream, eps=1e-5):
    result = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        result[idx] = (objective(layer, plus, upstream) - objective(layer, minus, upstream)) / (2 * eps)
    return result


@pytest.mark.parametrize(
    'batch,seq_len,embedding_dim,num_heads',
    [(1, 1, 4, 1), (2, 4, 8, 2), (2, 3, 12, 3), (1, 5, 16, 4)],
)
def test_forward_shapes(batch, seq_len, embedding_dim, num_heads):
    layer = make_layer(embedding_dim, num_heads)
    x = np.random.default_rng(1).normal(size=(batch, seq_len, embedding_dim)).astype(np.float32)
    output = layer.forward(x)
    head_dim = embedding_dim // num_heads
    assert output.shape == x.shape
    assert np.all(np.isfinite(output))
    for tensor in (layer.Q, layer.K, layer.V):
        assert tensor.shape == (batch, num_heads, seq_len, head_dim)
    assert layer.attention_weights.shape == (batch, num_heads, seq_len, seq_len)


@pytest.mark.parametrize('embedding_dim,num_heads', [(8, 0), (7, 2), (6, 2), (0, 1)])
def test_invalid_dimensions(embedding_dim, num_heads):
    with pytest.raises(ValueError):
        make_layer(embedding_dim, num_heads)


def test_causal_weights_and_normalization():
    layer = make_layer()
    x = np.random.default_rng(2).normal(size=(2, 5, 8)).astype(np.float32)
    layer.forward(x)
    weights = layer.attention_weights
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(weights >= 0)
    future_mask = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert np.all(weights[..., future_mask] == 0)
    np.testing.assert_allclose(weights[..., 0, 0], 1.0, atol=1e-6)


def test_future_tokens_do_not_affect_past_outputs():
    layer = make_layer()
    x = np.random.default_rng(3).normal(size=(2, 5, 8)).astype(np.float32)
    initial = layer.forward(x).copy()
    altered = x.copy()
    altered[:, 3:, :] += 50.0
    result = layer.forward(altered)
    np.testing.assert_allclose(result[:, :3], initial[:, :3], rtol=1e-5, atol=1e-5)


def test_rope_rotates_q_and_k_but_not_v():
    layer = make_layer()
    x = np.random.default_rng(4).normal(size=(2, 4, 8)).astype(np.float32)
    layer.forward(x)
    raw = layer.qkv.forward(x).reshape(2, 4, 3, 2, 4).transpose(2, 0, 3, 1, 4)
    raw_q, raw_k, raw_v = raw
    np.testing.assert_allclose(layer.V, raw_v, rtol=0, atol=0)
    np.testing.assert_allclose(layer.Q[..., 0, :], raw_q[..., 0, :], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(layer.K[..., 0, :], raw_k[..., 0, :], rtol=1e-6, atol=1e-6)
    # Rotations preserve each Q/K vector's L2 norm.
    np.testing.assert_allclose(np.linalg.norm(layer.Q, axis=-1), np.linalg.norm(raw_q, axis=-1), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(layer.K, axis=-1), np.linalg.norm(raw_k, axis=-1), rtol=1e-5, atol=1e-5)
    assert not np.allclose(layer.Q[..., 1:, :], raw_q[..., 1:, :])


def test_backward_shapes_and_finite_gradients():
    layer = make_layer(12, 3)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 4, 12)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    dx = layer.backward(upstream)
    assert dx.shape == x.shape
    assert np.all(np.isfinite(dx))
    for projection in (layer.qkv, layer.output_projection):
        assert projection.grad_weights.shape == projection.weights.shape
        assert projection.grad_bias.shape == projection.bias.shape
        assert np.all(np.isfinite(projection.grad_weights))
        assert np.all(np.isfinite(projection.grad_bias))


@pytest.mark.parametrize('embedding_dim,num_heads', [(4, 1), (8, 2), (12, 3)])
def test_numerical_input_gradient(embedding_dim, num_heads):
    layer = make_layer(embedding_dim, num_heads, seed=6)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 3, embedding_dim)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    layer.forward(x)
    analytic = layer.backward(upstream).copy()
    numerical = numerical_input_gradient(layer, x, upstream, eps=1e-5)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize('projection_name', ['qkv', 'output_projection'])
def test_numerical_weight_gradients(projection_name):
    layer = make_layer(4, 2, seed=8)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1, 3, 4)).astype(np.float64)
    upstream = rng.normal(size=x.shape).astype(np.float64)
    layer.forward(x)
    layer.backward(upstream)
    projection = getattr(layer, projection_name)
    analytic = projection.grad_weights.copy()
    eps = 1e-3  # Projection parameters use float32; larger epsilon reduces roundoff.
    numerical = np.zeros_like(projection.weights, dtype=np.float64)
    for idx in np.ndindex(projection.weights.shape):
        original = projection.weights[idx].item()
        projection.weights[idx] = original + eps
        plus = objective(layer, x, upstream)
        projection.weights[idx] = original - eps
        minus = objective(layer, x, upstream)
        projection.weights[idx] = original
        numerical[idx] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(analytic, numerical, rtol=6e-3, atol=6e-4)


def test_gradients_accumulate():
    layer = make_layer()
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 3, 8)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    layer.forward(x)
    layer.backward(upstream)
    original = [(p.grad_weights.copy(), p.grad_bias.copy()) for p in (layer.qkv, layer.output_projection)]
    layer.backward(upstream)
    for projection, (dw, db) in zip((layer.qkv, layer.output_projection), original):
        np.testing.assert_allclose(projection.grad_weights, 2 * dw, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(projection.grad_bias, 2 * db, rtol=1e-5, atol=1e-6)


def test_zero_grad():
    layer = make_layer()
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 3, 8)).astype(np.float32)
    layer.forward(x)
    layer.backward(np.ones_like(x))
    layer.zero_grad()
    for projection in (layer.qkv, layer.output_projection):
        assert np.all(projection.grad_weights == 0)
        assert np.all(projection.grad_bias == 0)


def test_parameters_expose_four_arrays_and_decay_flags():
    layer = make_layer()
    actual = layer.parameters()
    expected = [
        (layer.qkv.weights, layer.qkv.grad_weights, True),
        (layer.qkv.bias, layer.qkv.grad_bias, False),
        (layer.output_projection.weights, layer.output_projection.grad_weights, True),
        (layer.output_projection.bias, layer.output_projection.grad_bias, False),
    ]
    assert len(actual) == len(expected)
    for (w_a, g_a, decay_a), (w_e, g_e, decay_e) in zip(actual, expected):
        assert w_a is w_e
        assert g_a is g_e
        assert decay_a is decay_e

# python -m pytest tests/test_multi_head_attention.py -v