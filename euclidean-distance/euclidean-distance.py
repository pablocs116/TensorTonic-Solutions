import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    if len(x) != len(y):
        raise ValueError
        
    return np.sqrt(np.sum([(x_i-y_i)**2 for x_i, y_i in zip(x,y)]))