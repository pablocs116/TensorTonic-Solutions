import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    def nested(x):
        return len(x) > 0 and isinstance(x[0], (list, tuple))
        
    if nested(w):
        new_w, new_s = [], []
        for wi, gi, si in zip(w, g, s):
            nw, ns = rmsprop_step(wi, gi, si, lr, beta, eps)
            new_w.append(nw)
            new_s.append(ns)
        return new_w, new_s
        
    s = [beta * s_i + (1 - beta) * g_i**2 for s_i, g_i in zip(s, g)]
    w = [w_i - lr * g_i / (np.sqrt(s_i) + eps) for w_i, s_i, g_i in zip(w, s, g)]
    return w, s