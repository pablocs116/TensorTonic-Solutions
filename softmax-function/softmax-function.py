import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x)
    if x.ndim == 1:
        max = np.max(x)
        x = (np.exp(x-max))/(np.sum(np.exp(x - max)))
    else:
        max = [np.max(x_i) for x_i in x]
        x = [(np.exp(x_i-max_i))/(np.sum(np.exp(x_i - max_i))) for x_i, max_i in zip(x, max)]
        
    return x