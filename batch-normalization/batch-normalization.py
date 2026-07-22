import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    x = np.asarray(x)
    gamma = np.asarray(gamma)
    beta = np.asarray(beta)
    if x.ndim == 2:
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        norm = (x - mean)/np.sqrt(var + eps)
        
        return gamma * norm + beta
    else:
        mean = np.mean(x, axis = (0,2,3)).reshape(1, x.shape[1], 1, 1)
        var = np.var(x, axis = (0,2,3)).reshape(1, x.shape[1], 1, 1)
        gamma = gamma.reshape(1, x.shape[1], 1, 1)
        beta = beta.reshape(1, x.shape[1], 1, 1)
        norm = (x - mean)/np.sqrt(var + eps)
        
        return gamma * norm + beta