import numpy as np


class Rope:
    def __init__(self, head_dim:int, base:float= 10000.0):
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be positive and even")

        if not np.isfinite(base) or base <= 0:
            raise ValueError("base must be positive and finite") # no inf

        self.head_dim = head_dim
        self.base = base

        pair_indicies = np.arange(0, head_dim, 2) # steps every 2 from 0-head_dim
        self.frequencies = 1.0 / (base ** (pair_indicies / head_dim)) # 1.0 / theta ^ 2i/d, where 2i = pair_indicies


    def forward(self, x:np.ndarray):

        T = x.shape[-2] # number of tokens in each sequence 
        positions = np.arange(T) # ex: np.arange(4) -> [0,1,2,3]
        angles = positions[:, None] * self.frequencies[None, :] # theta p,i  = p*w i

        # sin & cos calculation (tells us how to rotate the pairs)
        sin = np.sin(angles)
        cos = np.cos(angles)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # rotation formula: xeven * cos(theta) - xodd*sin(theta) and xeven * sin(theta) + xodd*cos(theta)

        rotated_even = x_even * cos - x_odd*sin
        rotated_odd = x_even * sin + x_odd*cos

        output = np.empty_like(x)
        output[..., 0::2] = rotated_even
        output[...,1::2] = rotated_odd

        return output


    def backward(self, grad_output:np.ndarray):
        pass

if __name__ == "__main__":
    Q = np.array([
        [
            [1.0, 2.0, 3.0, 4.0],  # banana: position 0
            [2.0, 1.0, 4.0, 3.0]   # apple: position 1
        ]
    ])
    R = Rope(4)
    print(R.frequencies)
    print(R.forward(Q))






