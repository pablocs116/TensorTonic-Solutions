import numpy as np
def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    v = np.asarray(v, dtype=float)
    if v.ndim == 1:
        return float(np.linalg.norm(v))
    return np.asarray([float(np.linalg.norm(v_i)) for v_i in v])