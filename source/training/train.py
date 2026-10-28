"""Train a NumPy decoder-only GPT on the Shakespeare corpus.

Place at source/training/train.py, then run from project root:
    python -m training.train
    python -m training.train --resume
    python -m training.train --steps 100 --eval-interval 20 --no-show

Checkpoints store model parameters, AdamW moment buffers and step count,
plus RNG state for reproducible batch sampling.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model.gpt import GPT
from nn.loss import CrossEntropyLoss
from training.optimizer import AdamW

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.npy"
VAL_PATH = PROJECT_ROOT / "data" / "processed" / "val.npy"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoints" / "gpt_shakespeare_latest.npz"
HISTORY_PATH = OUTPUT_DIR / "training_history.npz"

VOCAB_SIZE = 1024
EMBEDDING_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
HIDDEN_DIM = 176
BATCH_SIZE = 4
CONTEXT_LENGTH = 64
STEPS = 2000
BASE_LR = 5e-4
MIN_LR = 5e-5
WARMUP_STEPS = 50
EVAL_INTERVAL = 100
EVAL_BATCHES = 5
SAVE_INTERVAL = 250
GRAD_CLIP_NORM = 1.0
SEED = 42


def sample_batch(tokens, batch_size, context_length, rng):
    """Random next-token windows; each call samples fresh start positions."""
    if tokens.ndim != 1 or len(tokens) <= context_length:
        raise ValueError("tokens must be a 1D array longer than context_length")
    starts = rng.integers(0, len(tokens) - context_length, size=batch_size)
    x = np.stack([tokens[i:i + context_length] for i in starts]).astype(np.int64)
    y = np.stack([tokens[i + 1:i + context_length + 1] for i in starts]).astype(np.int64)
    return x, y


def clip_gradients(parameters, max_norm):
    squared = sum(float(np.sum(grad.astype(np.float64) ** 2)) for _, grad, _ in parameters)
    norm = float(np.sqrt(squared))
    if not np.isfinite(norm):
        raise FloatingPointError("Non-finite gradients encountered")
    if norm > max_norm:
        factor = max_norm / (norm + 1e-12)
        for _, grad, _ in parameters:
            grad *= factor
    return norm


def learning_rate(step, total_steps):
    """Linear warmup followed by cosine decay. step is 1-indexed."""
    warmup = min(WARMUP_STEPS, max(1, total_steps // 10))
    if step <= warmup:
        return BASE_LR * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loss_fn, tokens, batch_size, context_length, n_batches, rng):
    """Validation only: never call backward or optimizer.step here."""
    losses = []
    for _ in range(n_batches):
        x, y = sample_batch(tokens, batch_size, context_length, rng)
        logits = model.forward(x)
        loss = float(loss_fn.forward(logits, y))
        if not np.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation loss")
        losses.append(loss)
    return float(np.mean(losses))


def save_checkpoint(path, model, optimizer, train_rng, next_step, config):
    """Save trainable weights + AdamW state; arrays are loaded by fixed index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"p_{i}": w.copy() for i, (w, _, _) in enumerate(model.parameters())}
    arrays.update({f"m_{i}": m.copy() for i, m in enumerate(optimizer.m)})
    arrays.update({f"v_{i}": v.copy() for i, v in enumerate(optimizer.v)})
    arrays["next_step"] = np.array(next_step, dtype=np.int64)
    arrays["optimizer_t"] = np.array(optimizer.t, dtype=np.int64)
    arrays["rng_state"] = np.array(json.dumps(train_rng.bit_generator.state))
    arrays["config"] = np.array(json.dumps(config, sort_keys=True))
    # Write through an open file so NumPy doesn't append '.npz' to a temp path.
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    temp.replace(path)


def load_checkpoint(path, model, optimizer, train_rng, config):
    """Restore parameters and optimizer state in place (preserves gradient refs)."""
    with np.load(path, allow_pickle=False) as checkpoint:
        stored_config = json.loads(str(checkpoint["config"].item()))
        if stored_config != config:
            raise ValueError(f"Checkpoint configuration mismatch: {stored_config} != {config}")
        parameters = model.parameters()
        for i, (w, _, _) in enumerate(parameters):
            saved = checkpoint[f"p_{i}"]
            if saved.shape != w.shape:
                raise ValueError(f"Parameter {i} shape mismatch")
            w[...] = saved
        for i in range(len(parameters)):
            optimizer.m[i][...] = checkpoint[f"m_{i}"]
            optimizer.v[i][...] = checkpoint[f"v_{i}"]
        optimizer.t = int(checkpoint["optimizer_t"].item())
        train_rng.bit_generator.state = json.loads(str(checkpoint["rng_state"].item()))
        return int(checkpoint["next_step"].item())


def save_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(HISTORY_PATH, **{key: np.asarray(values) for key, values in history.items()})


def load_history():
    if not HISTORY_PATH.exists():
        return None
    with np.load(HISTORY_PATH, allow_pickle=False) as data:
        return {key: data[key].tolist() for key in data.files}


def plot_history(history, show=True):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(history["train_steps"], history["train_loss"], label="Training batches", alpha=0.65)
    if history["val_steps"]:
        axes[0].plot(history["val_steps"], history["val_loss"], marker="o", label="Validation")
    axes[0].set(title="GPT Shakespeare — Training and Validation Loss", ylabel="Cross-entropy")
    axes[0].legend()

    axes[1].plot(history["train_steps"], history["grad_norm"])
    axes[1].axhline(GRAD_CLIP_NORM, linestyle="--", label="Clip threshold")
    axes[1].set(title="Gradient Norm Before Clipping", ylabel="L2 norm")
    axes[1].legend()

    axes[2].plot(history["train_steps"], history["lr"])
    axes[2].set(title="Learning Rate Schedule", xlabel="Optimizer step", ylabel="Learning rate")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    dest = PLOT_DIR / "full_training_metrics.png"
    fig.savefig(dest, dpi=160, bbox_inches="tight")
    print(f"Saved plots: {dest}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train NumPy GPT on Shakespeare")
    parser.add_argument("--steps", type=int, default=STEPS, help="Total optimization steps")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--eval-batches", type=int, default=EVAL_BATCHES)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.context_length, args.eval_interval,
           args.eval_batches, args.save_interval) < 1:
        parser.error("All numeric arguments must be positive")

    train_tokens = np.load(TRAIN_PATH, allow_pickle=False)
    val_tokens = np.load(VAL_PATH, allow_pickle=False)
    for split, tokens in (("train", train_tokens), ("validation", val_tokens)):
        if tokens.ndim != 1 or tokens.size <= args.context_length:
            raise ValueError(f"{split} data must be 1D and longer than context length")
        if np.min(tokens) < 0 or np.max(tokens) >= VOCAB_SIZE:
            raise ValueError(f"{split} token IDs exceed vocabulary size {VOCAB_SIZE}")

    train_rng = np.random.default_rng(SEED + 1)
    eval_rng = np.random.default_rng(SEED + 2)
    model_rng = np.random.default_rng(SEED)
    model = GPT(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                hidden_dim=HIDDEN_DIM, rng=model_rng)
    loss_fn = CrossEntropyLoss()
    params = model.parameters()
    optimizer = AdamW(parameters=params, lr=BASE_LR, weight_decay=0.01)
    config = dict(vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                  num_heads=NUM_HEADS, num_layers=NUM_LAYERS, hidden_dim=HIDDEN_DIM,
                  batch_size=args.batch_size, context_length=args.context_length,
                  total_steps=args.steps, base_lr=BASE_LR, min_lr=MIN_LR,
                  warmup_steps=WARMUP_STEPS, seed=SEED)

    history = {"train_steps": [], "train_loss": [], "grad_norm": [], "lr": [],
               "val_steps": [], "val_loss": []}
    start_step = 0
    if args.resume:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(CHECKPOINT_PATH)
        start_step = load_checkpoint(CHECKPOINT_PATH, model, optimizer, train_rng, config)
        history = load_history()
        if history is None:
            raise FileNotFoundError(f"Missing history needed for resume: {HISTORY_PATH}")
        print(f"Resumed checkpoint at completed step {start_step}")

    print(f"Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")
    print(f"Training steps: {args.steps} | Batch: {args.batch_size} | Context: {args.context_length}")
    for step in range(start_step + 1, args.steps + 1):
        x, y = sample_batch(train_tokens, args.batch_size, args.context_length, train_rng)
        model.zero_grad()
        logits = model.forward(x)
        loss = float(loss_fn.forward(logits, y))
        if not np.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step}")
        model.backward(loss_fn.backward())
        grad_norm = clip_gradients(params, GRAD_CLIP_NORM)
        lr = learning_rate(step, args.steps)
        optimizer.lr = lr
        optimizer.step()

        history["train_steps"].append(step)
        history["train_loss"].append(loss)
        history["grad_norm"].append(grad_norm)
        history["lr"].append(lr)

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            # Evaluation happens AFTER backward + update; layer caches may be overwritten safely.
            val_loss = evaluate(model, loss_fn, val_tokens, args.batch_size,
                                args.context_length, args.eval_batches, eval_rng)
            history["val_steps"].append(step)
            history["val_loss"].append(val_loss)
            print(f"Step {step:5d}/{args.steps} | train {loss:.4f} | val {val_loss:.4f} "
                  f"| grad {grad_norm:.3f} | lr {lr:.6f}", flush=True)
        elif step % 25 == 0:
            print(f"Step {step:5d}/{args.steps} | train {loss:.4f}", flush=True)

        if step % args.save_interval == 0 or step == args.steps:
            save_history(history)
            save_checkpoint(CHECKPOINT_PATH, model, optimizer, train_rng, step, config)
            print(f"Checkpoint saved at step {step}", flush=True)

    plot_history(history, show=not args.no_show)
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
