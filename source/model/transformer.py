from nn.rmsnorm import RMSNorm
from nn.multi_head_attention import MultiHeadAttention
from nn.mlp import SwiGLU

import numpy as np
class TransformerBlock:

    def __init__(self, embedding_dim:int, num_heads:int, hidden_dim:int, rng:np.random.Generator):
        self.norm1 = RMSNorm(embedding_dim)
        self.attention = MultiHeadAttention(embedding_dim, num_heads, rng)
        self.norm2 = RMSNorm(embedding_dim)
        self.mlp = SwiGLU(embedding_dim, hidden_dim, rng)



    def forward(self, x:np.ndarray):
        # attention + first residual connection
        h = x + self.attention.forward(self.norm1.forward(x))


        # swiglu + second residual connection
        y = h + self.mlp.forward(self.norm2.forward(h))
        return y

    def backward(self, grad_output:np.ndarray):
        grad_mlp_input = self.mlp.backward(grad_output)
        grad_h_mlp = self.norm2.backward(grad_mlp_input)
        grad_h = grad_output + grad_h_mlp


        # h = x + attention(norm1(x))

        grad_attention_input = self.attention.backward(grad_h)
        grad_x_attention = self.norm1.backward(grad_attention_input)

        grad_input = grad_h + grad_x_attention

        return grad_input


    def parameters(self):
        params = []

        for layer in (
            self.norm1, 
            self.attention,
            self.norm2,
            self.mlp
        ):
            params.extend(layer.parameters())

        return params

    def zero_grad(self):
            self.norm1.zero_grad()
            self.attention.zero_grad()
            self.norm2.zero_grad()
            self.mlp.zero_grad()