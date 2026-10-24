import numpy as np

class AdamW:
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0
        self.m = [np.zeros_like(w) for w, dw, decay in self.parameters]
        self.v = [np.zeros_like(w) for w, dw, decay in self.parameters]


    def step(self):
        """
        Performs a single optimization step.
        w: numpy array of weights
        dw: numpy array of gradients (same shape as w)
        """
            
        self.t += 1
        for i, (w, dw, decay) in enumerate(self.parameters):
            # 1. Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * dw
            
            # 2. Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (dw ** 2)
            
            # 3. Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # 4. Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            decay_update = self.weight_decay * w if decay else 0.0

            
            # 5. Apply gradient update + decoupled weight decay directly to weights
            # w = w - lr * (m_hat / (sqrt(v_hat) + eps)) - lr * weight_decay * w
            w -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + decay_update)
        
