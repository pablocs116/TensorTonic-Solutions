import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    dist = 0
    for i, j  in zip(x, y):
        dist += np.sqrt((i - j)**2)
    return dist