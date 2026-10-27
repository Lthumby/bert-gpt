"""Tests for single-head causal self-attention (before RoPE).

Run from the project root:
    python -m pytest tests/test_attention.py -v

Expected implementation: nn.attention.CausalSelfAttention, with forward(),
backward(), zero_grad(), and parameters().
"""

import numpy as np
import pytest

from nn.attention import CausalSelfAttention


def make_attention(embedding_dim=4, head_dim=4, seed=42):
    return CausalSelfAttention(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        rng=np.random.default_rng(seed),
    )


def scalar_objective(layer, x, upstream):
    """A scalar whose derivative is upstream through layer.forward(x)."""
    return float(np.sum(layer.forward(x) * upstream))


@pytest.mark.parametrize(
    "batch, seq_len, embedding_dim, head_dim",
    [(1, 1, 4, 4), (2, 3, 4, 4), (3, 5, 6, 2)],
)
def test_forward_output_shape_and_finite(batch, seq_len, embedding_dim, head_dim):
    layer = make_attention(embedding_dim, head_dim)
    x = np.random.default_rng(1).normal(
        size=(batch, seq_len, embedding_dim)
    ).astype(np.float32)

    output = layer.forward(x)

    assert output.shape == (batch, seq_len, head_dim)
    assert np.all(np.isfinite(output))
    assert layer.attention_weights.shape == (batch, seq_len, seq_len)


def test_attention_weights_are_causal_and_normalized():
    layer = make_attention()
    x = np.random.default_rng(2).normal(size=(2, 5, 4)).astype(np.float32)
    layer.forward(x)
    weights = layer.attention_weights

    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)
    assert np.all(weights >= 0)
    future_positions = np.triu(np.ones((5, 5), dtype=bool), k=1)
    assert np.all(weights[:, future_positions] == 0)
    np.testing.assert_allclose(weights[:, 0, 0], 1.0, atol=1e-7)


def test_future_tokens_cannot_change_past_outputs():
    layer = make_attention()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 4, 4)).astype(np.float32)

    baseline = layer.forward(x).copy()
    modified = x.copy()
    modified[:, 2:, :] += 100.0
    changed = layer.forward(modified)

    # The first two outputs may only depend on the first two inputs.
    np.testing.assert_allclose(changed[:, :2, :], baseline[:, :2, :], atol=1e-6)


def test_first_output_equals_first_value_projection():
    layer = make_attention()
    x = np.random.default_rng(4).normal(size=(2, 4, 4)).astype(np.float32)
    output = layer.forward(x)

    # At position 0, the mask forces 100% attention to position 0.
    np.testing.assert_allclose(output[:, 0, :], layer.V[:, 0, :], atol=1e-6)


def test_backward_shape_and_finite_gradients():
    layer = make_attention(embedding_dim=6, head_dim=4)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, 6)).astype(np.float32)
    upstream = rng.normal(size=(2, 3, 4)).astype(np.float32)

    layer.forward(x)
    grad_input = layer.backward(upstream)

    assert grad_input.shape == x.shape
    assert np.all(np.isfinite(grad_input))
    for projection in (layer.query, layer.key, layer.value):
        assert projection.grad_weights.shape == projection.weights.shape
        assert projection.grad_bias.shape == projection.bias.shape
        assert np.all(np.isfinite(projection.grad_weights))
        assert np.all(np.isfinite(projection.grad_bias))


def test_numerical_input_gradient():
    """Check dL/dX against central finite differences."""
    layer = make_attention(embedding_dim=3, head_dim=2, seed=10)
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 3, 3)).astype(np.float64)
    upstream = rng.normal(size=(1, 3, 2)).astype(np.float64)

    layer.forward(x)
    analytic = layer.backward(upstream).copy()

    eps = 1e-4
    numerical = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus, minus = x.copy(), x.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical[index] = (
            scalar_objective(layer, plus, upstream)
            - scalar_objective(layer, minus, upstream)
        ) / (2 * eps)

    np.testing.assert_allclose(analytic, numerical, rtol=3e-3, atol=3e-4)


@pytest.mark.parametrize("projection_name", ["query", "key", "value"])
def test_numerical_projection_weight_gradients(projection_name):
    """Independently verify gradients of Q, K and V projection weights."""
    layer = make_attention(embedding_dim=3, head_dim=2, seed=12)
    rng = np.random.default_rng(13)
    x = rng.normal(size=(1, 3, 3)).astype(np.float32)
    upstream = rng.normal(size=(1, 3, 2)).astype(np.float32)

    layer.forward(x)
    layer.backward(upstream)
    projection = getattr(layer, projection_name)
    analytic = projection.grad_weights.copy()

    eps = 1e-3  # Weights are float32; use a larger perturbation.
    numerical = np.zeros_like(projection.weights)
    for index in np.ndindex(projection.weights.shape):
        original = projection.weights[index].item()
        projection.weights[index] = original + eps
        loss_plus = scalar_objective(layer, x, upstream)
        projection.weights[index] = original - eps
        loss_minus = scalar_objective(layer, x, upstream)
        projection.weights[index] = original
        numerical[index] = (loss_plus - loss_minus) / (2 * eps)

    np.testing.assert_allclose(analytic, numerical, rtol=5e-3, atol=5e-4)


def test_backward_accumulates_gradients():
    layer = make_attention()
    rng = np.random.default_rng(14)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=(1, 3, 4)).astype(np.float32)
    layer.forward(x)

    layer.backward(upstream)
    first = [
        (p.grad_weights.copy(), p.grad_bias.copy())
        for p in (layer.query, layer.key, layer.value)
    ]
    layer.backward(upstream)

    for projection, (dw, db) in zip((layer.query, layer.key, layer.value), first):
        np.testing.assert_allclose(projection.grad_weights, 2 * dw, atol=1e-6)
        np.testing.assert_allclose(projection.grad_bias, 2 * db, atol=1e-6)


def test_zero_grad_resets_all_projection_gradients():
    layer = make_attention()
    rng = np.random.default_rng(15)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=(1, 3, 4)).astype(np.float32)
    layer.forward(x)
    layer.backward(upstream)

    layer.zero_grad()

    for projection in (layer.query, layer.key, layer.value):
        assert np.all(projection.grad_weights == 0)
        assert np.all(projection.grad_bias == 0)


def test_parameters_expose_six_arrays_with_decay_flags():
    layer = make_attention()
    params = layer.parameters()
    assert len(params) == 6

    expected = [
        (layer.query.weights, layer.query.grad_weights, True),
        (layer.query.bias, layer.query.grad_bias, False),
        (layer.key.weights, layer.key.grad_weights, True),
        (layer.key.bias, layer.key.grad_bias, False),
        (layer.value.weights, layer.value.grad_weights, True),
        (layer.value.bias, layer.value.grad_bias, False),
    ]
    for (actual_w, actual_dw, actual_decay), (w, dw, decay) in zip(params, expected):
        assert actual_w is w
        assert actual_dw is dw
        assert actual_decay is decay
    