import numpy as np

def _filter(y, i):
    return np.sum(y == i) / len(y)

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    if len(y) == 0:
        return 0.0
    
    H = 0
    C = np.unique(y)

    for i in C:
        p = _filter(y, i)
        if p > 0:
            H -= p * np.log2(p)

    return H