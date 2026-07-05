import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    if scores.ndim == 2:
        n, m = scores.shape
        new_scores = scores.copy()
    
        for i in range(n):
            for j in range(i+1,m):
                new_scores[i][j] = mask_value
    elif scores.ndim == 3:
        n, m, h = scores.shape
        new_scores = scores.copy()
        for i in range(n):
            for j in range(m):
                for z in range(j+1,h):
                    new_scores[i][j][z] = mask_value

    else:
        n, m, h, t = scores.shape
        new_scores = scores.copy()
        for i in range(n):
            for j in range(m):
                for z in range(h):
                    for v in range(z+1,t):
                        new_scores[i][j][z][v] = mask_value
        
    return new_scores