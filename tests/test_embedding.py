
import numpy as np
import pytest

from nn.embedding import Embedding


# ==========================================
# Initialization Tests
# ==========================================

def test_embedding_initialization():
    rng = np.random.default_rng(42)
    embedding = Embedding(1024, 64, rng)

    assert embedding.weights.shape == (1024, 64)
    assert embedding.weights.dtype == np.float32
    assert embedding.grad_weights.shape == (1024, 64)
    assert np.all(embedding.grad_weights == 0)


def test_embedding_reproducibility():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    emb1 = Embedding(1024, 64, rng1)
    emb2 = Embedding(1024, 64, rng2)

    np.testing.assert_array_equal(emb1.weights, emb2.weights)


# ==========================================
# Forward Pass Tests
# ==========================================

def test_forward_single_sequence():
    rng = np.random.default_rng(42)
    embedding = Embedding(1024, 64, rng)

    token_ids = np.array([2, 0, 2])
    output = embedding.forward(token_ids)

    assert output.shape == (3, 64)
    assert output.dtype == np.float32

    np.testing.assert_array_equal(output[0], embedding.weights[2])
    np.testing.assert_array_equal(output[1], embedding.weights[0])
    np.testing.assert_array_equal(output[2], embedding.weights[2])


def test_forward_batch():
    rng = np.random.default_rng(42)
    embedding = Embedding(1024, 64, rng)

    token_ids = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    output = embedding.forward(token_ids)

    assert output.shape == (2, 3, 64)

    np.testing.assert_array_equal(output[0, 0], embedding.weights[1])
    np.testing.assert_array_equal(output[1, 2], embedding.weights[6])


# ==========================================
# Backward Pass Tests
# ==========================================

def test_backward_gradient_accumulation():
    rng = np.random.default_rng(42)
    embedding = Embedding(4, 3, rng)

    token_ids = np.array([2, 0, 2])
    embedding.forward(token_ids)

    grad_output = np.array([
        [0.1, 0.2, 0.3],
        [0.4, 0.1, 0.2],
        [0.2, 0.3, 0.1]
    ], dtype=np.float32)

    embedding.backward(grad_output)

    expected = np.array([
        [0.4, 0.1, 0.2],
        [0.0, 0.0, 0.0],
        [0.3, 0.5, 0.4],
        [0.0, 0.0, 0.0]
    ], dtype=np.float32)

    np.testing.assert_allclose(
        embedding.grad_weights,
        expected,
        rtol=1e-6,
        atol=1e-7
    )


def test_backward_batch():
    rng = np.random.default_rng(42)
    embedding = Embedding(5, 2, rng)

    token_ids = np.array([
        [1, 2],
        [1, 3]
    ])

    embedding.forward(token_ids)

    grad_output = np.array([
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6], [0.7, 0.8]]
    ], dtype=np.float32)

    embedding.backward(grad_output)

    expected = np.array([
        [0.0, 0.0],
        [0.6, 0.8],
        [0.3, 0.4],
        [0.7, 0.8],
        [0.0, 0.0]
    ], dtype=np.float32)

    np.testing.assert_allclose(
        embedding.grad_weights, expected, atol=1e-7
    )


def test_backward_does_not_modify_weights():
    rng = np.random.default_rng(42)
    embedding = Embedding(4, 3, rng)

    original_weights = embedding.weights.copy()

    token_ids = np.array([0, 2])
    embedding.forward(token_ids)

    grad_output = np.ones((2, 3), dtype=np.float32)
    embedding.backward(grad_output)

    np.testing.assert_array_equal(
        embedding.weights, original_weights
    )


def test_backward_accumulates_across_calls():
    rng = np.random.default_rng(42)
    embedding = Embedding(4, 3, rng)

    token_ids = np.array([1, 2])
    embedding.forward(token_ids)

    grad_output = np.ones((2, 3), dtype=np.float32)

    embedding.backward(grad_output)
    embedding.backward(grad_output)

    np.testing.assert_array_equal(
        embedding.grad_weights[1],
        np.array([2.0, 2.0, 2.0])
    )

    np.testing.assert_array_equal(
        embedding.grad_weights[2],
        np.array([2.0, 2.0, 2.0])
    )


# ==========================================
# Gradient Reset Test
# ==========================================

def test_zero_grad():
    rng = np.random.default_rng(42)
    embedding = Embedding(4, 3, rng)

    embedding.forward(np.array([1, 2]))
    embedding.backward(np.ones((2, 3), dtype=np.float32))

    embedding.zero_grad()

    assert np.all(embedding.grad_weights == 0)
def test_invalid_token_id():
    embedding = Embedding(4, 3, np.random.default_rng(42))

    with pytest.raises(ValueError):
        embedding.forward(np.array([0, 5]))


def test_negative_token_id():
    embedding = Embedding(4, 3, np.random.default_rng(42))

    with pytest.raises(ValueError):
        embedding.forward(np.array([0, -1]))


def test_float_token_ids():
    embedding = Embedding(4, 3, np.random.default_rng(42))

    with pytest.raises(TypeError):
        embedding.forward(np.array([0.0, 1.0]))


def test_wrong_gradient_shape():
    embedding = Embedding(4, 3, np.random.default_rng(42))

    embedding.forward(np.array([0, 2]))

    with pytest.raises(ValueError):
        embedding.backward(np.ones((2, 5), dtype=np.float32))


def test_backward_before_forward():
    embedding = Embedding(4, 3, np.random.default_rng(42))

    with pytest.raises(RuntimeError):
        embedding.backward(np.ones((2, 3), dtype=np.float32))

#python -m pytest tests/test_embedding.py -v