"""Tests for a pre-norm TransformerBlock implemented with NumPy.

Save as tests/test_transformer_block.py and run from the project root:
    python -m pytest tests/test_transformer_block.py -v

Expected API:
    model.transformer_block.TransformerBlock(embedding_dim, num_heads, hidden_dim, rng)
    .forward(x), .backward(grad_output), .parameters(), .zero_grad()

The block is expected to compute:
    h = x + attention(norm1(x))
    y = h + mlp(norm2(h))
"""

import numpy as np
import pytest

from model.transformer import TransformerBlock


def make_block(embedding_dim=4, num_heads=2, hidden_dim=8, seed=42):
    return TransformerBlock(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        rng=np.random.default_rng(seed),
    )


def objective(block, x, upstream):
    """Scalar objective, L = sum(forward(x) * upstream)."""
    return float(np.sum(block.forward(x) * upstream))


def finite_difference_array(array, evaluate, eps=1e-3):
    """Central finite differences, suitable for float32 parameters."""
    grad = np.zeros_like(array, dtype=np.float64)
    for index in np.ndindex(array.shape):
        original = array[index].item()
        try:
            array[index] = original + eps
            plus = evaluate()
            array[index] = original - eps
            minus = evaluate()
        finally:
            array[index] = original
        grad[index] = (plus - minus) / (2.0 * eps)
    return grad


@pytest.mark.parametrize(
    "shape,num_heads,hidden_dim",
    [((1, 1, 4), 2, 8), ((2, 3, 4), 2, 12), ((2, 4, 8), 4, 16)],
)
def test_forward_shape_and_finite(shape, num_heads, hidden_dim):
    block = make_block(shape[-1], num_heads, hidden_dim)
    x = np.random.default_rng(1).normal(size=shape).astype(np.float32)
    y = block.forward(x)
    assert y.shape == shape
    assert np.all(np.isfinite(y))


def test_forward_matches_explicit_residual_equations():
    block = make_block()
    x = np.random.default_rng(2).normal(size=(2, 3, 4)).astype(np.float32)
    y = block.forward(x)
    h = x + block.attention.forward(block.norm1.forward(x))
    expected = h + block.mlp.forward(block.norm2.forward(h))
    np.testing.assert_allclose(y, expected, rtol=1e-6, atol=1e-6)


def test_future_tokens_cannot_change_earlier_outputs():
    block = make_block()
    x = np.random.default_rng(3).normal(size=(1, 5, 4)).astype(np.float32)
    reference = block.forward(x).copy()
    changed = x.copy()
    changed[:, 3:, :] += 20.0
    result = block.forward(changed)
    np.testing.assert_allclose(result[:, :3], reference[:, :3], rtol=1e-6, atol=1e-5)


def test_batch_examples_are_independent():
    block = make_block()
    x = np.random.default_rng(4).normal(size=(2, 3, 4)).astype(np.float32)
    reference = block.forward(x).copy()
    changed = x.copy()
    changed[1] += 4.0
    result = block.forward(changed)
    np.testing.assert_allclose(result[0], reference[0], rtol=1e-6, atol=1e-6)


def test_backward_shapes_and_finiteness():
    block = make_block(embedding_dim=8, num_heads=2, hidden_dim=12)
    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 3, 8)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    block.forward(x)
    dx = block.backward(upstream)
    assert dx.shape == x.shape
    assert np.all(np.isfinite(dx))
    for weights, grad, decay in block.parameters():
        assert weights.shape == grad.shape
        assert np.all(np.isfinite(grad))
        assert isinstance(decay, (bool, np.bool_))


def test_numerical_input_gradient():
    block = make_block(embedding_dim=4, num_heads=2, hidden_dim=6, seed=6)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    block.forward(x)
    analytical = block.backward(upstream).copy()
    numerical = finite_difference_array(x, lambda: objective(block, x, upstream))
    np.testing.assert_allclose(analytical, numerical, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "layer_name,param_name",
    [
        ("norm1", "gamma"),
        ("norm2", "gamma"),
        ("attention.qkv", "weights"),
        ("attention.qkv", "bias"),
        ("attention.output_projection", "weights"),
        ("mlp.gate", "weights"),
        ("mlp.up", "weights"),
        ("mlp.down", "weights"),
    ],
)
def test_numerical_parameter_gradient(layer_name, param_name):
    block = make_block(embedding_dim=4, num_heads=2, hidden_dim=6, seed=8)
    rng = np.random.default_rng(9)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    block.forward(x)
    block.backward(upstream)
    layer = block
    for attribute in layer_name.split("."):
        layer = getattr(layer, attribute)
    weights = getattr(layer, param_name)
    grad_name = "grad_gamma" if param_name == "gamma" else "grad_" + param_name
    analytical = getattr(layer, grad_name).copy()
    numerical = finite_difference_array(weights, lambda: objective(block, x, upstream))
    np.testing.assert_allclose(analytical, numerical, rtol=2e-2, atol=2e-3)


def test_zero_grad_clears_every_parameter_gradient():
    block = make_block()
    rng = np.random.default_rng(10)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    block.forward(x)
    block.backward(np.ones_like(x))
    assert any(np.any(grad != 0) for _, grad, _ in block.parameters())
    block.zero_grad()
    for _, grad, _ in block.parameters():
        assert np.all(grad == 0)


def test_backward_accumulates_parameter_gradients():
    block = make_block()
    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 3, 4)).astype(np.float32)
    upstream = rng.normal(size=x.shape).astype(np.float32)
    block.forward(x)
    block.backward(upstream)
    first = [grad.copy() for _, grad, _ in block.parameters()]
    block.backward(upstream)
    for (_, grad, _), previous in zip(block.parameters(), first):
        np.testing.assert_allclose(grad, 2 * previous, rtol=1e-5, atol=1e-6)


def test_parameters_lists_each_submodule_once():
    block = make_block()
    expected = (
        block.norm1.parameters()
        + block.attention.parameters()
        + block.norm2.parameters()
        + block.mlp.parameters()
    )
    params = block.parameters()
    assert len(params) == len(expected) == 12
    for (w, grad, decay), (ew, eg, ed) in zip(params, expected):
        assert w is ew
        assert grad is eg
        assert decay == ed
    assert len({id(w) for w, _, _ in params}) == len(params)


def test_optimizer_parameter_decay_flags():
    block = make_block()
    params = block.parameters()
    # Both RMSNorm scale vectors and all biases should avoid weight decay.
    assert sum(bool(decay) for _, _, decay in params) == 5
    assert sum(not bool(decay) for _, _, decay in params) == 7

    
# python -m pytest tests/test_transformer.py -v