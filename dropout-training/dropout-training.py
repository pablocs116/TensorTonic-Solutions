import numpy as np

def dropout(x, p=0.5, rng=None):
    if rng is None:
        rng = np.random
        
    x = np.asarray(x, dtype=float)
    scale = 1.0 / (1.0 - p)
    keep = rng.random(x.shape) >= p
    out = x * keep * scale
    return out, keep.astype(float)*scale  