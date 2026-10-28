import numpy as np
from nn.linear import Linear



class SwiGLU:

    def __init__(self, embedding_dim:int, hidden_dim:int, rng:np.random.Generator):
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Dimensions must be positive")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.gate = Linear(embedding_dim, hidden_dim, rng)
        self.up = Linear(embedding_dim, hidden_dim, rng)
        self.down = Linear(hidden_dim, embedding_dim, rng)


    def forward(self, x:np.ndarray):
        gate = self.gate.forward(x)
        up = self.up.forward(x) 


        sigmoid = 1.0 / (1.0 + np.exp(-gate))
        activated_gate = gate * sigmoid
        hidden = activated_gate * up

        output = self.down.forward(hidden)

        self.up_vals = up
        self.activated_gate = activated_gate
        self.sigmoid = sigmoid
        self.gate_vals = gate
        return output

    def backward(self, grad_output:np.ndarray):
        grad_hidden = self.down.backward(grad_output)
        grad_activated_gate = grad_hidden * self.up_vals
        grad_up = grad_hidden * self.activated_gate

        silu_derivative = (
            self.sigmoid + self.gate_vals * self.sigmoid * (1.0 - self.sigmoid)
        )
        grad_gate = grad_activated_gate * silu_derivative

        grad_x_gate = self.gate.backward(grad_gate)
        grad_x_up = self.up.backward(grad_up)
        grad_input = grad_x_gate + grad_x_up

        return grad_input   

    def zero_grad(self):
        self.gate.zero_grad()
        self.up.zero_grad()
        self.down.zero_grad()

    def parameters(self):   
        params=[]

        for layer in (self.gate, self.up, self.down):
            params.extend([
                (layer.weights, layer.grad_weights, True),
                (layer.bias, layer.grad_bias, False)
            ])
        return params