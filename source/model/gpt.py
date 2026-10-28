from model.transformer import TransformerBlock as tb
from nn.rmsnorm import RMSNorm
from nn.embedding import Embedding
from nn.linear import Linear
import numpy as np

class GPT:

    def __init__(self, vocab_size:int, embedding_dim:int, rng: np.random.Generator, num_heads:int=4, num_layers:int=2, hidden_dim:int=352):
        self.embedding = Embedding(vocab_size, embedding_dim, rng)

        self.blocks= [
            tb(embedding_dim=embedding_dim,num_heads=num_heads, hidden_dim=hidden_dim, rng=rng) for _ in range(num_layers)
        ]

        self.final_norm = RMSNorm(embedding_dim)


        self.linear = Linear(embedding_dim, vocab_size, rng)



    def forward(self, token_ids:np.ndarray):
        x = self.embedding.forward(token_ids)
        for block in self.blocks:
            x = block.forward(x)
        x = self.final_norm.forward(x)
        logits = self.linear.forward(x)
        return logits

    def backward(self, grad_output: np.ndarray):
        grad= self.linear.backward(grad_output)
        grad = self.final_norm.backward(grad)

        for block in reversed(self.blocks):
            grad = block.backward(grad)

        self.embedding.backward(grad)

    def parameters(self):
        params = [
            (self.embedding.weights, self.embedding.grad_weights, True)
        ]

        for block in self.blocks:
            params.extend(block.parameters())

        params.extend(self.final_norm.parameters())

        params.extend([
            (self.linear.weights, self.linear.grad_weights, True),
            (self.linear.bias, self.linear.grad_bias, False)
        ])

        return params

    
    def zero_grad(self):
        self.embedding.zero_grad()

        for block in self.blocks:
            block.zero_grad()

        self.final_norm.zero_grad()
        self.linear.zero_grad()


if __name__ == "__main__":
    rng = np.random.default_rng(42)

    model = GPT(
        vocab_size=1024,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        hidden_dim=352,
        rng=rng
    )

    token_ids = rng.integers(
        0, 1024, size=(2, 16)
    )

    logits = model.forward(token_ids)

    print("Input shape:", token_ids.shape)
    print("Output shape:", logits.shape)