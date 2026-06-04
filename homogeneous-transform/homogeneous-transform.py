import numpy as np

def apply_homogeneous_transform(T, points):
    points = np.asarray(points, dtype=float)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, -1)
    ones = np.ones((points.shape[0], 1))
    homog = np.hstack([points[:, :3], ones])          # N×4
    out = (T @ homog.T).T                             # N×4
    out = out[:, :3]                 
    return out[0] if single else out