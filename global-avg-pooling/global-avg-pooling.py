import numpy as np

def global_avg_pool(x):
    """(C,H,W) => (C,)  |  (N,C,H,W) => (N,C)"""
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
    return np.mean(x, axis=(-2, -1))