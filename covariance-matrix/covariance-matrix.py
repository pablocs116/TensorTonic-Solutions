import numpy as np

def covariance_matrix(X):
    X = np.array(X, dtype=float)
    if X.ndim == 1:
        return None
    
    n = X.shape[0]
    if n < 2:
        return None
    
    mu = np.mean(X, axis=0)
    X_cent = X - mu
    return (X_cent.T @ X_cent) / (n - 1)