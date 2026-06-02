import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    y = np.asarray(y).reshape(-1)
    i = 0
    
    while i < steps:
        z = X @ w + b
        p = _sigmoid(z)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        dw = (1 / m) * (X.T @ (p - y))
        db = (1 / m) * np.sum(p - y)
        w -= lr * dw
        b -= lr * db
        i+=1
    return (w, b)