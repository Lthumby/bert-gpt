import numpy as np


class CrossEntropyLoss:
    def __init__(self):
       pass

    def forward(self, logits:np.ndarray, targets:np.ndarray): # -> single scalar loss

        self.targets = targets.copy()   
        self.orig_shape = logits.shape
        logits = logits - np.max(logits, axis=-1, keepdims=True) # [1,2,3] -> [-2, 1, 0] stabalizes

        # Soft-Max
        exp_logit = np.exp(logits) # e ^ zi

        sum_exp_logit = np.sum(exp_logit, axis=-1, keepdims=True) # sum(e ^ zi)

        soft_prob = exp_logit / sum_exp_logit
        self.soft_prob = soft_prob.copy()   

        logits_2d = logits.reshape(-1, logits.shape[-1]) # flatten logits
        targs_1d = targets.reshape(-1) # flatten targets

        target_logits = logits_2d[np.arange(len(targs_1d)), targs_1d]
        log_sum_exp = np.log(sum_exp_logit.reshape(-1))

        loss = np.mean(log_sum_exp - target_logits)

        return loss

    def backward(self):

        # ( P - Y )/ N
        grad_logits = self.soft_prob.copy()

        gradlog_2d = grad_logits.reshape(-1, grad_logits.shape[-1]) # flattens from (#, #, #) -> (#*#, #)
        targs_1d = self.targets.reshape(-1) # flattens

        # P-Y calc
        gradlog_2d[np.arange(len(targs_1d)), targs_1d] -= 1


        grad_logits /= len(targs_1d)

        return grad_logits



