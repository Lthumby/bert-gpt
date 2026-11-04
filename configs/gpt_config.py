
from dataclasses import dataclass, asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 1024
    embedding_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    hidden_dim: int = 352
    context_length: int = 128

    # Training
    batch_size: int = 8
    max_steps: int = 10000

    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 300

    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0

    # Evaluation and checkpoints
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 1000

    # Reproducibility
    seed: int = 42

    # Files
    train_path: str = str(
        PROJECT_ROOT / "data/processed/train.npy"
    )
    val_path: str = str(
        PROJECT_ROOT / "data/processed/val.npy"
    )
    checkpoint_dir: str = str(
        PROJECT_ROOT / "outputs/checkpoints"
    )
    plot_dir: str = str(
        PROJECT_ROOT / "outputs/plots"
    )

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        if (self.embedding_dim // self.num_heads) % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE"
            )

        if self.num_layers <= 0 or self.hidden_dim <= 0:
            raise ValueError(
                "num_layers and hidden_dim must be positive"
            )

        if self.context_length <= 0 or self.batch_size <= 0:
            raise ValueError(
                "context_length and batch_size must be positive"
            )

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

        if not (0 < self.min_lr <= self.peak_lr):
            raise ValueError("Invalid learning-rate range")

        if not (0 <= self.warmup_steps < self.max_steps):
            raise ValueError("Invalid warmup configuration")

        if self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("Invalid optimization configuration")

        if min(
            self.eval_interval,
            self.eval_batches,
            self.save_interval
        ) <= 0:
            raise ValueError("Evaluation intervals must be positive")

    @property
    def head_dim(self) -> int:
        return self.embedding_dim // self.num_heads

    def to_dict(self) -> dict:
        return asdict(self)
