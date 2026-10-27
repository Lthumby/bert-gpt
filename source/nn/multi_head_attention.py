import numpy as np

from nn.attention import CausalSelfAttention as CSA
from nn.linear import Linear
from nn.rope import RoPE


class MultiHeadAttention:

    def __init__(self, embedding_dim:int, num_heads:int, rng:np.random.Generator):
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads # floor division

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.qkv = Linear(embedding_dim, 3*embedding_dim, rng) # concatenates layers 

        # Create CSA instance for each head 
        self.heads = [CSA(embedding_dim=embedding_dim, head_dim=self.head_dim, rng=rng) for _ in range(num_heads)]
        self.output_projection = Linear(embedding_dim, embedding_dim, rng)

        self.rope = RoPE(self.head_dim)


    def forward(self, x:np.ndarray):
        qkv = self.qkv.forward(x)
        B,T,D = x.shape

        qkv = qkv.reshape(B,T,3,self.num_heads, self.head_dim) # Batch, Tokens, qkv, heads, features

        qkv = qkv.transpose(2, 0, 3, 1, 4) # 3, B, H, T, Dh
        Q,K,V = qkv[0], qkv[1], qkv[2]

        Q = self.rope.forward(Q)
        K = self.rope.forward(K)

        scores = Q @ np.swapaxes(K, -1, -2)
        scores = scores / np.sqrt(self.head_dim)

        mask = np.tril(np.ones((T,T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)
        # softmax
        stable_scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(stable_scores)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        context = attention_weights @ V

        context = context.transpose(0,2,1,3)
        context = context.reshape(B, T, D)

        output = self.output_projection.forward(context)
        return output


    def backward(self, grad_output:np.ndarray):
        pass





