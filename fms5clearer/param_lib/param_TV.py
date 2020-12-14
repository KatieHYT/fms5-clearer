import numpy as np
class params():
    def __init__(self, mu1 = 4, mu2 = 30, mu3 = 60):
        self.mu1 = mu1
        self.mu2 = mu2
        self.mu3 = mu3
        self.relchg_cond = 1e-3
        self.beta1 = 128
        self.beta2 = 128
        self.gamma = 1.618
        self.mini_iter_nonsat = 30
        self.mini_iter_sat = np.floor(self.mini_iter_nonsat/4)
        
        self.max_iter = 100
        self.Guide = 2
        self.L2ratio = 2^4
