import numpy as np

def streaming_minmax_init(D):
    """
    Initialize state dict with min, max arrays of shape (D,).
    """
    # Write code here
    return {
        "max": [-np.inf]*D,
        "min": [np.inf]*D
    }

def streaming_minmax_update(state, X_batch, eps=1e-8):
    """
    Update state's min/max with X_batch, return normalized batch.
    """
    X_batch = np.asarray(X_batch, dtype=float)

    # update running min/max element-wise over the batch
    state["min"] = np.minimum(state["min"], X_batch.min(axis=0))
    state["max"] = np.maximum(state["max"], X_batch.max(axis=0))

    # normalize with the *updated* stats
    return (X_batch - state["min"]) / (state["max"] - state["min"] + eps)