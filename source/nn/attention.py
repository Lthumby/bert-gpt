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

        # Scores = (QK^t) / sqrt(d k)

        k_transposed = np.swapaxes(K, -1, -2) # -1: last dim, -2: second to last dim

        scores_top = Q @ k_transposed

        scores = scores_top / np.sqrt(self.head_dim)

        # mask
        T = x.shape[1] # sequence length
        mask = np.tril(np.ones((T,T), dtype=bool))
        scores = np.where(mask, scores, -np.inf)

        # attention softmax

        stable_scores = scores - np.max(scores, axis=-1, keepdims=True)

        exp_scores = np.exp(stable_scores)

        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        output = attention_weights @ V

        return output



if __name__ == "__main__":
    pass