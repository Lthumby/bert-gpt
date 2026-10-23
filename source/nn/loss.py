import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, logits:np.ndarray, targets:np.ndarray): # -> single scalar loss

        self.targets = targets.copy()
        logits = logits - np.max(logits, axis=-1, keepdims=True) # [1,2,3] -> [-2, 1, 0] stabalizes

        targ_arr = np.array([])
        # Soft-Max
        exp_logit = np.exp(logits) # e ^ zi

        sum_exp_logit = np.sum(exp_logit, axis=-1, keepdims=True) # sum(e ^ zi)

        soft_prob = exp_logit / sum_exp_logit
        self.soft_prob = soft_prob.copy()   

        probs_2d = soft_prob.reshape(-1, soft_prob.shape[-1])
        targs_1d = targets.reshape(-1) # dims are 3d, so fix to 2d to use np.arange

        targ_arr = probs_2d[np.arange(len(targs_1d)), targs_1d] 

        targ_arr = -np.log(targ_arr)

    
        return np.mean(targ_arr)


    def backward(self):

        # ( P - Y )/ N
        grad_logits = self.soft_prob.copy()

        gradlog_2d = grad_logits.reshape(-1, grad_logits.shape[-1]) # flattens from (#, #, #) -> (#*#, #)
        targs_1d = self.targets.reshape(-1) # flattens

        # P-Y calc
        gradlog_2d[np.arange(len(targs_1d)), targs_1d] -= 1


        grad_logits /= len(targs_1d)

        return grad_logits





if __name__ == "__main__":
    l = CrossEntropyLoss()

    probabilities = np.array([
    [0.65, 0.25, 0.10],
    [0.15, 0.70, 0.15],
    [0.20, 0.30, 0.50]

    ])  
    ph = np.array([])

    targets = np.array([0, 2, 1])

    tp = np.add.at(ph, targets, max(probabilities))

    print(tp)