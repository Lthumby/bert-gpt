
import numpy as np
import pytest

from nn.linear import Linear


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def linear():
    rng = np.random.default_rng(42)
    return Linear(input_dim=3, output_dim=2, rng=rng)


# ==========================================
# Initialization Tests
# ==========================================

def test_initialization_shapes(linear):
    assert linear.weights.shape == (3, 2)
    assert linear.bias.shape == (2,)
    assert linear.grad_weights.shape == (3, 2)
    assert linear.grad_bias.shape == (2,)


def test_initialization_dtype(linear):
    assert linear.weights.dtype == np.float32
    assert linear.bias.dtype == np.float32
    assert linear.grad_weights.dtype == np.float32
    assert linear.grad_bias.dtype == np.float32


def test_initial_gradients_are_zero(linear):
    assert np.all(linear.grad_weights == 0)
    assert np.all(linear.grad_bias == 0)


def test_reproducible_initialization():
    layer1 = Linear(3, 2, np.random.default_rng(42))
    layer2 = Linear(3, 2, np.random.default_rng(42))

    np.testing.assert_array_equal(
        layer1.weights, layer2.weights
    )


# ==========================================
# Forward Pass Tests
# ==========================================

def test_forward_known_values(linear):
    linear.weights[:] = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=np.float32)

    linear.bias[:] = [0.5, 1.0]

    x = np.array([[2, 3, 4]], dtype=np.float32)

    output = linear.forward(x)

    expected = np.array([[31.5, 41.0]], dtype=np.float32)

    np.testing.assert_allclose(output, expected)


def test_forward_3d_shape():
    layer = Linear(64, 128, np.random.default_rng(42))

    x = np.ones((16, 128, 64), dtype=np.float32)
    output = layer.forward(x)

    assert output.shape == (16, 128, 128)
    assert output.dtype == np.float32


# ==========================================
# Backward Pass Tests
# ==========================================

def test_backward_known_values(linear):
    linear.weights[:] = np.array([
        [1, 2],
        [3, 4],
        [5, 6]
    ], dtype=np.float32)

    x = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ], dtype=np.float32)

    grad_output = np.array([
        [1, 2],
        [3, 4]
    ], dtype=np.float32)

    linear.forward(x)
    grad_input = linear.backward(grad_output)

    expected_grad_weights = np.array([
        [13, 18],
        [17, 24],
        [21, 30]
    ], dtype=np.float32)

    expected_grad_bias = np.array([4, 6], dtype=np.float32)

    expected_grad_input = np.array([
        [5, 11, 17],
        [11, 25, 39]
    ], dtype=np.float32)

    np.testing.assert_allclose(
        linear.grad_weights, expected_grad_weights
    )

    np.testing.assert_allclose(
        linear.grad_bias, expected_grad_bias
    )

    np.testing.assert_allclose(
        grad_input, expected_grad_input
    )


def test_backward_3d_shape():
    layer = Linear(64, 128, np.random.default_rng(42))

    x = np.ones((2, 4, 64), dtype=np.float32)
    grad_output = np.ones((2, 4, 128), dtype=np.float32)

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    assert grad_input.shape == (2, 4, 64)
    assert layer.grad_weights.shape == (64, 128)
    assert layer.grad_bias.shape == (128,)

    np.testing.assert_allclose(
        layer.grad_weights,
        np.full((64, 128), 8, dtype=np.float32)
    )

    np.testing.assert_allclose(
        layer.grad_bias,
        np.full(128, 8, dtype=np.float32)
    )


def test_backward_accumulates_gradients(linear):
    x = np.ones((2, 3), dtype=np.float32)
    grad_output = np.ones((2, 2), dtype=np.float32)

    linear.forward(x)

    linear.backward(grad_output)

    first_grad_weights = linear.grad_weights.copy()
    first_grad_bias = linear.grad_bias.copy()

    linear.backward(grad_output)

    np.testing.assert_allclose(
        linear.grad_weights, first_grad_weights * 2
    )

    np.testing.assert_allclose(
        linear.grad_bias, first_grad_bias * 2
    )


def test_backward_does_not_modify_parameters(linear):
    original_weights = linear.weights.copy()
    original_bias = linear.bias.copy()

    x = np.ones((2, 3), dtype=np.float32)
    linear.forward(x)
    linear.backward(np.ones((2, 2), dtype=np.float32))

    np.testing.assert_array_equal(
        linear.weights, original_weights
    )

    np.testing.assert_array_equal(
        linear.bias, original_bias
    )


# ==========================================
# Gradient Reset Test
# ==========================================

def test_zero_grad(linear):
    x = np.ones((2, 3), dtype=np.float32)

    linear.forward(x)
    linear.backward(np.ones((2, 2), dtype=np.float32))

    linear.zero_grad()

    assert np.all(linear.grad_weights == 0)
    assert np.all(linear.grad_bias == 0)


# ==========================================
# Numerical Gradient Tests
# ==========================================

def test_numerical_weight_gradients(linear):
    x = np.array([
        [0.2, -0.3, 0.5],
        [0.1, 0.4, -0.2]
    ], dtype=np.float32)

    grad_output = np.array([
        [0.7, -0.2],
        [-0.4, 0.3]
    ], dtype=np.float32)

    linear.forward(x)
    linear.backward(grad_output)

    analytic_grad = linear.grad_weights.copy()

    # Use float64 parameters for a more reliable
    # finite-difference gradient check.
    linear.weights = linear.weights.astype(np.float64)
    linear.bias = linear.bias.astype(np.float64)

    epsilon = 1e-5
    numerical_grad = np.zeros_like(linear.weights)

    for i in range(linear.weights.shape[0]):
        for j in range(linear.weights.shape[1]):
            original = linear.weights[i, j]

            linear.weights[i, j] = original + epsilon
            loss_plus = np.sum(
                linear.forward(x) * grad_output
            )

            linear.weights[i, j] = original - epsilon
            loss_minus = np.sum(
                linear.forward(x) * grad_output
            )

            linear.weights[i, j] = original

            numerical_grad[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    np.testing.assert_allclose(
        analytic_grad,
        numerical_grad,
        rtol=1e-4,
        atol=1e-5
    )


def test_numerical_input_gradients(linear):
    x = np.array([
        [0.2, -0.3, 0.5],
        [0.1, 0.4, -0.2]
    ], dtype=np.float64)

    linear.weights = linear.weights.astype(np.float64)
    linear.bias = linear.bias.astype(np.float64)

    grad_output = np.array([
        [0.7, -0.2],
        [-0.4, 0.3]
    ], dtype=np.float64)

    linear.forward(x)
    analytic_grad = linear.backward(grad_output)

    epsilon = 1e-5
    numerical_grad = np.zeros_like(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            original = x[i, j]

            x[i, j] = original + epsilon
            loss_plus = np.sum(
                linear.forward(x) * grad_output
            )

            x[i, j] = original - epsilon
            loss_minus = np.sum(
                linear.forward(x) * grad_output
            )

            x[i, j] = original

            numerical_grad[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    np.testing.assert_allclose(
        analytic_grad,
        numerical_grad,
        rtol=1e-4,
        atol=1e-5
    )
# python -m pytest tests/test_linear.py -v