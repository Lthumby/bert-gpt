import numpy as np


class RMSNorm:

    def __init__(self, embedding_dim:int, eps:float=1e-6):
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive int")
        if not np.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")

        self.embedding_dim = embedding_dim
        self.eps = eps

        self.gamma = np.ones(embedding_dim, dtype=np.float32) # scaling parameter

        self.grad_gamma = np.zeros_like(self.gamma) # accumlated grad for optimizer

    def forward(self, x:np.ndarray):
        # y = xi / sqrt(x^2 + eps)
        mean_square = np.mean(x ** 2, axis=-1, keepdims=True)

        inv_rms = 1.0 / np.sqrt(mean_square + self.eps)
        normalized = x * inv_rms
        output = normalized * self.gamma

        self.inv_rms = inv_rms
        self.normalized = normalized
        self.x = x
        return output
    
    def backward(self, grad_output:np.ndarray):
        self.grad_gamma += np.sum(grad_output * self.normalized, axis=(0,1))
        grad_normalized = grad_output * self.gamma
        dot = np.sum(grad_normalized * self.x, axis=-1, keepdims=True)

        grad_input = (grad_normalized * self.inv_rms - (self.x * self.inv_rms**3 / self.embedding_dim) * dot)

        return grad_input 



if __name__ == "__main__":
    rmsnorm = RMSNorm(4)

    x = np.array(
        [[[2.0, 4.0, 6.0, 8.0]]],
        dtype=np.float32
    )

    output = rmsnorm.forward(x)

    print("Input:", x)
    print("Output:", output)
    print("Gamma:", rmsnorm.gamma)