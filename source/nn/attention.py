import numpy as np
from nn.linear import Linear 

class CausalSelfAttention:
    def __init__(self, embedding_dim:int, head_dim:int, rng:np.random.Generator):
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim

        self.query = Linear(embedding_dim, head_dim, rng)
        self.key = Linear(embedding_dim, head_dim, rng)
        self.value = Linear(embedding_dim, head_dim, rng)


    def forward(self, x:np.ndarray): 
        Q = self.query.forward(x)
        K = self.key.forward(x)
        V = self.value.forward(x)

        self.K = K
        self.Q = Q
        # Scores = (QK^t) / sqrt(d k)

        k_transposed = np.swapaxes(K, -1, -2) # -1: last dim, -2: second to last dim

        scores_top = Q @ k_transposed

        scores = scores_top / np.sqrt(self.head_dim)

        # mask
        T = x.shape[1] # sequence length
        mask = np.tril(np.ones((T,T), dtype=bool))
        self.mask = mask
        scores = np.where(mask, scores, -np.inf)

        # attention softmax

        stable_scores = scores - np.max(scores, axis=-1, keepdims=True)

        exp_scores = np.exp(stable_scores)

        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self.attention_weights = attention_weights
        self.V = V

        output = attention_weights @ V

        return output

    def backward(self, grad_output):
        grad_V = (
            np.swapaxes(self.attention_weights, -1, -2) @ grad_output
        )

        grad_attention = (
            grad_output @ np.swapaxes(self.V, -1, -2)
        )

        P = self.attention_weights
        dP = grad_attention

        weighted_grad = P * dP

        row_sum = np.sum(weighted_grad,axis=-1, keepdims=True)
        grad_scores = P * (dP - row_sum)

        grad_scores = np.where(self.mask, grad_scores, 0) # further enforcment to not calculate 0's

        raw_grad = grad_scores / np.sqrt(self.head_dim)

        grad_Q = raw_grad @ self.K

        grad_K = np.swapaxes(raw_grad, -1,-2) @ self.Q

        grad_x_Q = self.query.backward(grad_Q)
        grad_x_K = self.key.backward(grad_K)
        grad_x_V = self.value.backward(grad_V)

        grad_input = grad_x_Q + grad_x_K + grad_x_V

        return grad_input

    def zero_grad(self):
        self.query.zero_grad()
        self.key.zero_grad()
        self.value.zero_grad()

    def parameters(self): # for adamW
        return [
            (self.query.weights, self.query.grad_weights, True),
            (self.query.bias, self.query.grad_bias, False),

            (self.key.weights, self.key.grad_weights, True),
            (self.key.bias, self.key.grad_bias, False),

            (self.value.weights, self.value.grad_weights, True),
            (self.value.bias, self.value.grad_bias, False),
        ]



if __name__ == "__main__":
    pass