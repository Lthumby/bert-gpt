
import numpy as np
import pytest

from dataset.dataset import TextDataset
from dataset.batching import get_batch


# ==========================================
# TextDataset Tests
# ==========================================

def test_dataset_initialization():
    tokens = np.arange(20)
    dataset = TextDataset(tokens, block_size=3)

    assert np.array_equal(dataset.token_arr, tokens)
    assert dataset.block_size == 3


def test_dataset_length():
    tokens = np.arange(20)
    dataset = TextDataset(tokens, block_size=3)

    assert len(dataset) == 17


def test_first_training_example():
    tokens = np.array([10, 20, 30, 40, 50, 60])
    dataset = TextDataset(tokens, block_size=3)

    x, y = dataset[0]

    np.testing.assert_array_equal(x, [10, 20, 30])
    np.testing.assert_array_equal(y, [20, 30, 40])


def test_last_training_example():
    tokens = np.array([10, 20, 30, 40, 50, 60])
    dataset = TextDataset(tokens, block_size=3)

    x, y = dataset[len(dataset) - 1]

    np.testing.assert_array_equal(x, [30, 40, 50])
    np.testing.assert_array_equal(y, [40, 50, 60])


def test_target_shift():
    tokens = np.arange(100)
    dataset = TextDataset(tokens, block_size=8)

    for i in range(len(dataset)):
        x, y = dataset[i]

        np.testing.assert_array_equal(x[1:], y[:-1])
        assert len(x) == len(y) == 8


# ==========================================
# Batching Tests
# ==========================================

def test_batch_shapes():
    tokens = np.arange(100)
    dataset = TextDataset(tokens, block_size=8)

    rng = np.random.default_rng(42)

    x, y = get_batch(dataset, batch_size=16, rng=rng)

    assert x.shape == (16, 8)
    assert y.shape == (16, 8)


def test_batch_target_shift():
    tokens = np.arange(100)
    dataset = TextDataset(tokens, block_size=8)

    rng = np.random.default_rng(42)
    x, y = get_batch(dataset, batch_size=16, rng=rng)

    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])
    np.testing.assert_array_equal(y[:, -1], x[:, -1] + 1)


def test_batch_reproducibility():
    tokens = np.arange(100)
    dataset = TextDataset(tokens, block_size=8)

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    x1, y1 = get_batch(dataset, batch_size=16, rng=rng1)
    x2, y2 = get_batch(dataset, batch_size=16, rng=rng2)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_batch_contains_valid_examples():
    tokens = np.arange(100)
    dataset = TextDataset(tokens, block_size=8)

    rng = np.random.default_rng(42)
    x_batch, y_batch = get_batch(dataset, batch_size=16, rng=rng)

    # Every row should correspond to a valid dataset window.
    for x, y in zip(x_batch, y_batch):
        start = int(x[0])

        expected_x, expected_y = dataset[start]

        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)
# python -m pytest tests/test_dataset.py -v 