"""Tests for source/nn/rope.py (adjacent-pair RoPE).

Run from project root:
    python -m pytest tests/test_rope.py -v

Assumes your project exposes: from nn.rope import RoPE
"""

import numpy as np
import pytest

from nn.rope import RoPE


def numerical_input_gradient(rope, x, upstream, epsilon=1e-6):
    """Central differences for L(x) = sum(RoPE(x) * upstream)."""
    numerical = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += epsilon
        x_minus[idx] -= epsilon

        loss_plus = np.sum(rope.forward(x_plus) * upstream)
        loss_minus = np.sum(rope.forward(x_minus) * upstream)
        numerical[idx] = (loss_plus - loss_minus) / (2 * epsilon)
    return numerical


@pytest.mark.parametrize("shape", [(1, 2, 4), (2, 3, 8), (2, 2, 3, 4)])
def test_backward_matches_numerical_gradient(shape):
    """Works for both (B,T,D) and (B,H,T,D) inputs."""
    rng = np.random.default_rng(42)
    rope = RoPE(head_dim=shape[-1])
    x = rng.normal(size=shape).astype(np.float64)
    upstream = rng.normal(size=shape).astype(np.float64)

    # Analytic gradient from RoPE's custom backward method.
    rope.forward(x)
    analytic = rope.backward(upstream)

    # Independent finite-difference gradient from forward only.
    numerical = numerical_input_gradient(rope, x, upstream)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-5, atol=1e-6)


def test_backward_known_rotation():
    """At position 1, first pair is rotated by 1 radian."""
    rope = RoPE(head_dim=4)
    x = np.zeros((1, 2, 4), dtype=np.float64)
    rope.forward(x)

    upstream = np.zeros_like(x)
    upstream[0, 1, :2] = [0.5, 0.2]
    analytic = rope.backward(upstream)

    expected = np.zeros_like(x)
    expected[0, 1, 0] = 0.5 * np.cos(1.0) + 0.2 * np.sin(1.0)
    expected[0, 1, 1] = -0.5 * np.sin(1.0) + 0.2 * np.cos(1.0)
    np.testing.assert_allclose(analytic, expected, atol=1e-12)


def test_backward_preserves_gradient_norm():
    """Orthogonal rotations preserve gradient L2 norm."""
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 5, 8))
    upstream = rng.normal(size=x.shape)
    rope = RoPE(head_dim=8)
    rope.forward(x)
    grad = rope.backward(upstream)
    np.testing.assert_allclose(
        np.linalg.norm(grad, axis=-1),
        np.linalg.norm(upstream, axis=-1),
        rtol=1e-12, atol=1e-12,
    )


def test_position_zero_is_identity_forward_and_backward():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(2, 4, 6))
    upstream = rng.normal(size=x.shape)
    rope = RoPE(head_dim=6)
    y = rope.forward(x)
    grad = rope.backward(upstream)
    np.testing.assert_allclose(y[:, 0], x[:, 0], atol=1e-12)
    np.testing.assert_allclose(grad[:, 0], upstream[:, 0], atol=1e-12)


def test_nontrivial_position_changes_vector():
    rope = RoPE(head_dim=4)
    x = np.ones((1, 2, 4), dtype=np.float64)
    y = rope.forward(x)
    assert not np.allclose(y[:, 1], x[:, 1])
