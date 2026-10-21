import numpy as np

SCALE = 0.02
class Embedding:
    def __init__(self, vocab_size:int, embedding_dim:int, rng:np.random.Generator):
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer")

        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.weights = rng.normal(
            loc=0.0,
            scale=SCALE,
            size=(vocab_size, embedding_dim)
        ).astype(np.float32)
        self.grad_weights = np.zeros_like(self.weights)
    

    def forward(self, token_ids:np.ndarray):

        if not isinstance(token_ids, np.ndarray):
            raise TypeError("token_ids must be a NumPy array")

        if not np.issubdtype(token_ids.dtype, np.integer):
            raise TypeError("token_ids must contain integers")

        if np.any(token_ids < 0) or np.any(token_ids >= self.vocab_size):
            raise ValueError("token_ids contains an invalid token ID")
        
        self.token_ids = token_ids.copy()
        return self.weights[token_ids]


    def zero_grad(self):
        self.grad_weights.fill(0)

    def backward(self, grad_output: np.ndarray):
        # goal: calculate goal_weights
        if not hasattr(self, "token_ids"):
            raise RuntimeError("forward() must be called before backward()")

        if not isinstance(grad_output, np.ndarray):
            raise TypeError("grad_output must be a NumPy array")

        if not np.issubdtype(grad_output.dtype, np.floating):
            raise TypeError("grad_output must contain floating-point values")

        expected_shape = self.token_ids.shape + (self.embedding_dim,)

        if grad_output.shape != expected_shape:
            raise ValueError(
                f"Expected grad_output shape {expected_shape}, "
                f"received {grad_output.shape}"
            )

        if not np.all(np.isfinite(grad_output)):
            raise ValueError("grad_output contains NaN or infinity")
        np.add.at(self.grad_weights, self.token_ids, grad_output)

