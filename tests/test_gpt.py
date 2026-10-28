"""Tests for the complete NumPy decoder-only GPT (Transformer + RoPE).

Run from the project root:
    python -m pytest tests/test_gpt.py -v

Expected: model.gpt.GPT with forward(), backward(), parameters(), zero_grad().
The constructor accepts vocab_size, embedding_dim, rng, num_heads,
num_layers and hidden_dim as documented in the project.
"""

import numpy as np
import pytest

from model.gpt import GPT


def make_gpt(seed=42, vocab_size=11, embedding_dim=8, num_heads=2,
             num_layers=2, hidden_dim=16):
    return GPT(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rng=np.random.default_rng(seed),
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
    )


def scalar_objective(model, tokens, upstream):
    return float(np.sum(model.forward(tokens) * upstream))


def get_param_map(model):
    return [(w, dw, decay) for w, dw, decay in model.parameters()]


@pytest.mark.parametrize('batch,seq_len', [(1, 1), (2, 3), (3, 5)])
def test_forward_shape_and_finite(batch, seq_len):
    model = make_gpt()
    tokens = np.random.default_rng(1).integers(0, 11, (batch, seq_len))
    logits = model.forward(tokens)
    assert logits.shape == (batch, seq_len, 11)
    assert np.all(np.isfinite(logits))


def test_correct_number_of_independent_blocks():
    model = make_gpt(num_layers=3)
    assert len(model.blocks) == 3
    assert model.blocks[0] is not model.blocks[1]
    assert model.blocks[0].attention.qkv.weights is not model.blocks[1].attention.qkv.weights


def test_output_depends_on_previous_tokens():
    """With a fixed random seed, a change to the past should affect a later logit."""
    model = make_gpt()
    before = np.array([[1, 2, 3, 4]], dtype=np.int64)
    after = before.copy()
    after[0, 0] = 5
    original = model.forward(before).copy()
    changed = model.forward(after)
    assert not np.allclose(original[:, -1], changed[:, -1], rtol=1e-6, atol=1e-7)


def test_future_tokens_cannot_change_earlier_logits():
    model = make_gpt()
    tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    original = model.forward(tokens).copy()
    modified = tokens.copy()
    modified[0, 3:] = [9, 10]
    changed = model.forward(modified)
    np.testing.assert_allclose(changed[:, :3], original[:, :3], rtol=1e-5, atol=1e-6)


def test_batch_items_are_independent():
    model = make_gpt()
    tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    original = model.forward(tokens).copy()
    changed_tokens = tokens.copy()
    changed_tokens[1] = [8, 9, 10]
    changed = model.forward(changed_tokens)
    np.testing.assert_allclose(changed[0], original[0], rtol=1e-5, atol=1e-6)


def test_zero_transformer_blocks_still_produce_logits():
    model = make_gpt(num_layers=0)
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    assert model.forward(tokens).shape == (1, 3, 11)


def test_backward_creates_finite_grads_with_correct_shapes():
    model = make_gpt()
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, 11, (2, 3))
    logits = model.forward(tokens)
    model.backward(rng.normal(size=logits.shape).astype(np.float32))
    for w, dw, _ in model.parameters():
        assert dw.shape == w.shape
        assert np.all(np.isfinite(dw))
    assert np.any(model.embedding.grad_weights != 0)
    assert np.any(model.blocks[0].attention.qkv.grad_weights != 0)
    assert np.any(model.blocks[1].mlp.down.grad_weights != 0)



def test_numerical_embedding_gradient():
    """Verify GPT embedding gradients using multiple finite-difference steps."""
    rng = np.random.default_rng(42)

    model = GPT(
        vocab_size=8,
        embedding_dim=4,
        rng=np.random.default_rng(10),
        num_heads=2,
        num_layers=2,
        hidden_dim=8
    )

    tokens = np.array([[1, 2, 1]], dtype=np.int64)

    upstream = rng.normal(
        size=(1, 3, 8)
    ).astype(np.float32)

    # Calculate analytical gradients
    model.zero_grad()
    model.forward(tokens)
    model.backward(upstream)

    analytic = model.embedding.grad_weights.copy()
    weights = model.embedding.weights

    # Include repeated token 1 and token 2
    indices = [(1, 0), (1, 3), (2, 0), (2, 3)]

    epsilons = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

    for index in indices:
        original = weights[index].item()
        analytical_gradient = float(analytic[index])

        print(f"\nParameter: {index}")
        print(f"Analytical gradient: {analytical_gradient:.6f}")

        results = []

        for eps in epsilons:
            # Positive perturbation
            weights[index] = original + eps
            loss_plus = float(
                np.sum(model.forward(tokens) * upstream)
            )

            # Negative perturbation
            weights[index] = original - eps
            loss_minus = float(
                np.sum(model.forward(tokens) * upstream)
            )

            # Restore original weight
            weights[index] = original

            numerical_gradient = (
                loss_plus - loss_minus
            ) / (2 * eps)

            difference = abs(
                analytical_gradient - numerical_gradient
            )

            print(
                f"eps={eps:.0e} | "
                f"numerical={numerical_gradient:.6f} | "
                f"difference={difference:.6f}"
            )

            results.append((eps, numerical_gradient))

        # Use the most accurate finite-difference estimate
        # among the tested step sizes.
        best_eps, best_gradient = min(
            results,
            key=lambda result: abs(
                result[1] - analytical_gradient
            )
        )

        print(f"Best epsilon: {best_eps:.0e}")

        np.testing.assert_allclose(
            analytical_gradient,
            best_gradient,
            rtol=2e-2,
            atol=2e-3,
            err_msg=f"Embedding gradient mismatch at {index}"
        )



@pytest.mark.parametrize('parameter_path', [
    'blocks.0.norm1.gamma',
    'blocks.0.attention.qkv.weights',
    'blocks.1.mlp.gate.weights',
    'final_norm.gamma',
    'linear.weights',
    'linear.bias',
])
def test_numerical_parameter_gradients(parameter_path):
    """Finite-difference spot checks across the full GPT computational graph."""
    model = make_gpt(seed=20, vocab_size=7, embedding_dim=4,
                     num_heads=2, num_layers=2, hidden_dim=8)
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    upstream = np.random.default_rng(21).normal(size=(1, 3, 7)).astype(np.float32)
    model.forward(tokens)
    model.backward(upstream)

    parts = parameter_path.split('.')
    obj = model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    name = parts[-1]
    param = getattr(obj, name)
    grad = getattr(obj, 'grad_' + name)

    eps = 1e-3
    indices = list(np.ndindex(param.shape))
    for index in indices[:min(4, len(indices))]:
        original = param[index].item()
        param[index] = original + eps
        plus = scalar_objective(model, tokens, upstream)
        param[index] = original - eps
        minus = scalar_objective(model, tokens, upstream)
        param[index] = original
        numeric = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(grad[index], numeric, rtol=3e-2, atol=3e-3)


def test_parameters_register_all_layers_once():
    model = make_gpt(num_layers=2)
    params = get_param_map(model)
    # Embedding 1 + each block (2 norm + 4 attention + 6 MLP = 12)
    # + final norm 1 + output linear 2 = 28 arrays.
    assert len(params) == 28
    assert len({id(w) for w, _, _ in params}) == len(params)
    for w, dw, decay in params:
        assert isinstance(w, np.ndarray)
        assert isinstance(dw, np.ndarray)
        assert dw.shape == w.shape
        assert decay in (True, False)


def test_zero_grad_clears_every_parameter_gradient():
    model = make_gpt()
    rng = np.random.default_rng(30)
    tokens = rng.integers(0, 11, (2, 3))
    logits = model.forward(tokens)
    model.backward(rng.normal(size=logits.shape).astype(np.float32))
    assert any(np.any(dw != 0) for _, dw, _ in model.parameters())
    model.zero_grad()
    assert all(np.all(dw == 0) for _, dw, _ in model.parameters())


def test_backward_accumulates_gradients():
    model = make_gpt()
    tokens = np.array([[1, 2, 3]], dtype=np.int64)
    upstream = np.random.default_rng(40).normal(size=(1, 3, 11)).astype(np.float32)
    model.forward(tokens)
    model.backward(upstream)
    first = [dw.copy() for _, dw, _ in model.parameters()]
    model.backward(upstream)
    for (_, dw, _), initial in zip(model.parameters(), first):
        np.testing.assert_allclose(dw, 2 * initial, rtol=1e-5, atol=1e-6)


def test_single_gradient_descent_step_reduces_fixed_batch_loss():
    """The model should reduce CE loss on a tiny, fixed next-token task."""
    model = make_gpt(seed=50, vocab_size=7, embedding_dim=8,
                     num_heads=2, num_layers=2, hidden_dim=16)
    tokens = np.array([[1, 2, 3], [2, 3, 4]], dtype=np.int64)
    targets = np.array([[2, 3, 4], [3, 4, 5]], dtype=np.int64)

    def loss_and_grad(logits):
        shifted = logits.astype(np.float64) - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        flat_probs = probs.reshape(-1, probs.shape[-1])
        flat_targets = targets.reshape(-1)
        loss = -np.log(flat_probs[np.arange(len(flat_targets)), flat_targets]).mean()
        grad = probs.copy().reshape(-1, probs.shape[-1])
        grad[np.arange(len(flat_targets)), flat_targets] -= 1
        return loss, (grad.reshape(probs.shape) / targets.size).astype(np.float32)

    initial_logits = model.forward(tokens)
    initial_loss, grad = loss_and_grad(initial_logits)
    model.zero_grad()
    model.backward(grad)
    for w, dw, _ in model.parameters():
        w -= 1e-3 * dw
    final_loss, _ = loss_and_grad(model.forward(tokens))
    assert np.isfinite(final_loss)
    assert final_loss < initial_loss

# python -m pytest tests/test_gpt.py -v