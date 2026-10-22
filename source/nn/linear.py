import numpy as np

class Linear:
    def __init__(self, input_dim:int, output_dim:int, rng:np.random.Generator):
        # Y = XW + b : X=inp_vec, W=weights, b=bias

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.SCALE = np.sqrt(2/(input_dim + output_dim)) # xavier
        self.weights = rng.normal( # trainable
                loc=0.0,
                scale=self.SCALE,
                size=(input_dim, output_dim)
            ).astype(np.float32)

        self.bias = np.zeros(output_dim, dtype=np.float32) # 1D array creation, trainable
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)


    def forward(self, arr_x:np.ndarray):
        self.arr_x = arr_x.copy()
        x_w = arr_x @ self.weights
        return x_w + self.bias

    def backward(self, grad_output:np.ndarray):
        # weight gradients (dW = x^T * dY)
        x_2d = self.arr_x.reshape(-1, self.input_dim)
        grad_2d = grad_output.reshape(-1, self.output_dim)
        self.grad_weights += x_2d.T @ grad_2d
        # bias gradients (db = sum(dY))
        self.grad_bias += np.sum(grad_output, axis=tuple(range(grad_output.ndim - 1)))
        # input gradients (dX = dY * W^T)
        grad_input = grad_output @ self.weights.T
        # return to prev. layer

        return grad_input

    def zero_grad(self):
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)
