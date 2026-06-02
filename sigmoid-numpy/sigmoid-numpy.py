import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
  Supports scalars, lists, nested lists, and numpy arrays.
    """
    if isinstance(x, (list, tuple)):
        return [sigmoid(i) for i in x]
    arr = np.asarray(x, dtype=float)
    out = 1.0 / (1.0 + np.exp(-arr))
    return out.item() if out.ndim == 0 else out