
import sys
import time
from pathlib import Path

import pytest
#RUN: PYTHONPATH=source python3 -m pytest tests/tokenizer_test.py -v -s
# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "source"
DATA_PATH = PROJECT_DIR / "data" / "raw" / "input.txt"

# Make source/ importable
sys.path.insert(0, str(SOURCE_DIR))

from tokenizer.tokenizer import Tokenizer


# =====================================
# FIXTURES
# =====================================

@pytest.fixture(scope="module")
def training_text():
    return DATA_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def trained_tokenizer(training_text):
    tokenizer = Tokenizer()

    start = time.perf_counter()
    tokenizer.train(training_text, vocab_size=1024)
    elapsed = time.perf_counter() - start

    print(f"\nTraining time: {elapsed:.2f}s")

    return tokenizer


# =====================================
# TEST 1: TRAINING
# =====================================

def test_vocab_size(trained_tokenizer):
    assert len(trained_tokenizer.vocab) == 1024


def test_learned_merge_count(trained_tokenizer):
    assert len(trained_tokenizer.merges) == 768


# =====================================
# TEST 2: SHAKESPEARE ROUND-TRIP
# =====================================

def test_shakespeare_round_trip(trained_tokenizer, training_text):
    sample = training_text[:1000]

    encoded = trained_tokenizer.encode(sample)
    decoded = trained_tokenizer.decode(encoded)

    assert decoded == sample


# =====================================
# TEST 3: SAVE AND LOAD
# =====================================

@pytest.fixture(scope="module")
def loaded_tokenizer(trained_tokenizer, tmp_path_factory):
    # Temporary file: won't overwrite your real merges.json
    filepath = tmp_path_factory.mktemp("tokenizer") / "merges.json"

    trained_tokenizer.save(filepath)

    tokenizer = Tokenizer()
    tokenizer.load(filepath)

    return tokenizer


def test_merges_preserved(trained_tokenizer, loaded_tokenizer):
    assert trained_tokenizer.merges == loaded_tokenizer.merges


def test_vocab_preserved(trained_tokenizer, loaded_tokenizer):
    assert trained_tokenizer.vocab == loaded_tokenizer.vocab


def test_encoding_consistency(
    trained_tokenizer,
    loaded_tokenizer,
    training_text
):
    sample = training_text[:1000]

    assert trained_tokenizer.encode(sample) == loaded_tokenizer.encode(sample)


# =====================================
# TEST 4: RETRAINING RESETS STATE
# =====================================

def test_retraining_resets_merges():
    tokenizer = Tokenizer()

    tokenizer.train("bananas", vocab_size=260)
    tokenizer.train("hello", vocab_size=258)

    fresh = Tokenizer()
    fresh.train("hello", vocab_size=258)

    assert tokenizer.merges == fresh.merges


def test_retraining_resets_vocab():
    tokenizer = Tokenizer()

    tokenizer.train("bananas", vocab_size=260)
    tokenizer.train("hello", vocab_size=258)

    fresh = Tokenizer()
    fresh.train("hello", vocab_size=258)

    assert tokenizer.vocab == fresh.vocab


# =====================================
# TEST 5: INVALID TOKEN IDS
# =====================================

def test_invalid_token_id(loaded_tokenizer):
    with pytest.raises(ValueError):
        loaded_tokenizer.decode([999999])


# =====================================
# TEST 6: EMPTY AND UNICODE TEXT
# =====================================

@pytest.mark.parametrize("text", [
    "",
    "hello",
    "café",
    "こんにちは",
    "Hello 🌎!",
    "😀🔥🚀",
    "To be, or not to be.",
    "ROMEO:\nO, she doth teach the torches to burn bright!"
])
def test_round_trip(loaded_tokenizer, text):
    encoded = loaded_tokenizer.encode(text)
    decoded = loaded_tokenizer.decode(encoded)

    assert decoded == text


# =====================================
# PERFORMANCE BENCHMARK
# =====================================

def test_full_dataset_performance(loaded_tokenizer, training_text):
    start = time.perf_counter()

    encoded = loaded_tokenizer.encode(training_text)

    elapsed = time.perf_counter() - start

    original_bytes = len(training_text.encode("utf-8"))
    encoded_tokens = len(encoded)

    print("\n=== TOKENIZER PERFORMANCE ===")
    print(f"Encoding time: {elapsed:.3f}s")
    print(f"Original bytes: {original_bytes:,}")
    print(f"BPE tokens: {encoded_tokens:,}")

    if encoded_tokens:
        print(f"Compression: {original_bytes / encoded_tokens:.2f}x")

    assert loaded_tokenizer.decode(encoded) == training_text
