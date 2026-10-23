
import numpy as np
import pytest

from nn.loss import CrossEntropyLoss


# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def loss_fn():
    return CrossEntropyLoss()


# ==========================================
# Forward Tests
# ==========================================

def test_forward_uniform_probabilities(loss_fn):
    """
    Equal logits produce equal probabilities.

    With 3 possible tokens, each probability is 1/3.
    Expected cross-entropy = log(3).
    """
    logits = np.zeros((2, 3), dtype=np.float32)
    targets = np.array([0, 2])

    loss = loss_fn.forward(logits, targets)

    np.testing.assert_allclose(
        loss, np.log(3), rtol=1e-6
    )


def test_forward_known_values(loss_fn):
    """
    Compare against a manually calculated loss.
    """
    logits = np.array([
        [2.0, 1.0, 0.1],
        [0.5, 2.0, 1.0]
    ], dtype=np.float32)

    targets = np.array([0, 2])

    # Independent reference calculation
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(
        np.sum(np.exp(shifted), axis=-1, keepdims=True)
    )

    expected = -np.mean([
        log_probs[0, 0],
        log_probs[1, 2]
    ])

    loss = loss_fn.forward(logits, targets)

    np.testing.assert_allclose(
        loss, expected, rtol=1e-6
    )


def test_forward_3d_logits(loss_fn):
    """
    Simulate GPT:
    (batch_size, sequence_length, vocab_size)
    """
    logits = np.zeros((2, 4, 5), dtype=np.float32)

    targets = np.array([
        [0, 1, 2, 3],
        [4, 3, 2, 1]
    ])

    loss = loss_fn.forward(logits, targets)

    np.testing.assert_allclose(
        loss, np.log(5), rtol=1e-6
    )


def test_forward_numerical_stability(loss_fn):
    """
    Large logits must not cause exponential overflow.
    """
    logits = np.array([
        [1000.0, 1001.0, 1002.0],
        [-1000.0, -1001.0, -1002.0]
    ], dtype=np.float32)

    targets = np.array([2, 0])

    loss = loss_fn.forward(logits, targets)

    assert np.isfinite(loss)
    assert loss >= 0


def test_forward_does_not_modify_logits(loss_fn):
    logits = np.array([
        [1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0]
    ], dtype=np.float32)

    original = logits.copy()
    targets = np.array([2, 0])

    loss_fn.forward(logits, targets)

    np.testing.assert_array_equal(logits, original)


def test_forward_extremely_unlikely_target(loss_fn):
    """
    A very unlikely correct token should still
    produce a finite, large loss.

    This test requires numerically stable loss
    calculation, such as log-sum-exp.
    """
    logits = np.array([
        [0.0, -1000.0, -2000.0]
    ], dtype=np.float64)

    targets = np.array([2])

    loss = loss_fn.forward(logits, targets)

    np.testing.assert_allclose(
        loss, 2000.0, rtol=1e-10
    )


# ==========================================
# Backward Tests
# ==========================================

def test_backward_uniform_probabilities(loss_fn):
    logits = np.zeros((1, 3), dtype=np.float32)
    targets = np.array([1])

    loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    expected = np.array([
        [1/3, -2/3, 1/3]
    ], dtype=np.float32)

    np.testing.assert_allclose(
        grad, expected, rtol=1e-6
    )


def test_backward_known_values(loss_fn):
    """
    Compare P - Y divided by N.
    """
    probabilities = np.array([
        [0.2, 0.6, 0.2],
        [0.1, 0.3, 0.6]
    ], dtype=np.float64)

    logits = np.log(probabilities)
    targets = np.array([1, 0])

    loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    expected = np.array([
        [0.10, -0.20, 0.10],
        [-0.45, 0.15, 0.30]
    ])

    np.testing.assert_allclose(
        grad, expected, rtol=1e-6
    )


def test_backward_3d_shape(loss_fn):
    logits = np.zeros((2, 4, 5), dtype=np.float32)
    targets = np.zeros((2, 4), dtype=np.int64)

    loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    assert grad.shape == logits.shape

    # There are 8 predictions total.
    expected = np.full(
        (2, 4, 5), 0.2 / 8, dtype=np.float32
    )
    expected[:, :, 0] = -0.8 / 8

    np.testing.assert_allclose(
        grad, expected, rtol=1e-6
    )


def test_backward_row_sums_zero(loss_fn):
    """
    Each prediction's logit gradients should sum
    to approximately zero.
    """
    rng = np.random.default_rng(42)

    logits = rng.normal(size=(3, 4, 5))
    targets = rng.integers(0, 5, size=(3, 4))

    loss_fn.forward(logits, targets)
    grad = loss_fn.backward()

    np.testing.assert_allclose(
        np.sum(grad, axis=-1),
        np.zeros((3, 4)),
        atol=1e-12
    )


def test_backward_does_not_modify_cached_probabilities(loss_fn):
    logits = np.array([
        [1.0, 2.0, 3.0]
    ])

    targets = np.array([2])

    loss_fn.forward(logits, targets)

    original = loss_fn.soft_prob.copy()

    loss_fn.backward()

    np.testing.assert_array_equal(
        loss_fn.soft_prob, original
    )


def test_backward_repeatable(loss_fn):
    """
    Repeated backward calls should return the
    same gradients, not accumulate them.
    """
    logits = np.array([
        [1.0, 2.0, 3.0]
    ])

    targets = np.array([2])

    loss_fn.forward(logits, targets)

    first = loss_fn.backward()
    second = loss_fn.backward()

    np.testing.assert_allclose(
        first, second, rtol=1e-12, atol=1e-12
    )


# ==========================================
# Numerical Gradient Test
# ==========================================

def test_numerical_gradients(loss_fn):
    """
    Compare analytic backward gradients against
    finite-difference approximations.
    """
    logits = np.array([
        [0.2, -0.3, 0.5],
        [0.1, 0.4, -0.2]
    ], dtype=np.float64)

    targets = np.array([2, 1])

    loss_fn.forward(logits, targets)
    analytic_grad = loss_fn.backward()

    epsilon = 1e-5
    numerical_grad = np.zeros_like(logits)

    for i in range(logits.shape[0]):
        for j in range(logits.shape[1]):
            plus = logits.copy()
            minus = logits.copy()

            plus[i, j] += epsilon
            minus[i, j] -= epsilon

            loss_plus = loss_fn.forward(plus, targets)
            loss_minus = loss_fn.forward(minus, targets)

            numerical_grad[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    np.testing.assert_allclose(
        analytic_grad,
        numerical_grad,
        rtol=1e-5,
        atol=1e-6
    )
# python -m pytest tests/test_loss.py -v