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

        self.attention_weights = attention_weights
        self.Q = Q
        self.V = V
        self.K = K
        self.mask = mask
        return output


    def backward(self, grad_output:np.ndarray):
        grad_context = self.output_projection.backward(grad_output)
        B,T,D = grad_context.shape
        grad_context = grad_context.reshape(B,T, self.num_heads, self.head_dim)
        grad_context = grad_context.transpose(0,2, 1, 3)

        grad_V = (
            np.swapaxes(self.attention_weights, -1, -2) @ grad_context
        )

        grad_attention = (
            grad_context @ np.swapaxes(self.V, -1, -2)
        )

        P = self.attention_weights
        row_sum = np.sum(P * grad_attention,axis=-1, keepdims=True)

        grad_scores = P * (grad_attention - row_sum)
        grad_scores = np.where(self.mask, grad_scores, 0.0)

        raw_grad = grad_scores / np.sqrt(self.head_dim)

        grad_Q = raw_grad @ self.K
        grad_K = np.swapaxes(raw_grad, -1,-2) @ self.Q

        grad_Q = self.rope.backward(grad_Q)
        grad_K = self.rope.backward(grad_K)

        grad_qkv = np.stack([grad_Q, grad_K, grad_V], axis=0)
        grad_qkv = grad_qkv.transpose(1, 3, 0, 2, 4)
        grad_qkv = grad_qkv.reshape(B,T, 3*D) # back to original

        grad_input = self.qkv.backward(grad_qkv)

        return grad_input

    def zero_grad(self):
        self.qkv.zero_grad()
        self.output_projection.zero_grad()

    def parameters(self):
        return [
            (self.qkv.weights, self.qkv.grad_weights, True),
            (self.qkv.bias, self.qkv.grad_bias, False),
            (
                self.output_projection.weights,
                self.output_projection.grad_weights,
                True
            ),
            (
                self.output_projection.bias,
                self.output_projection.grad_bias,
                False
            ),
        ]



