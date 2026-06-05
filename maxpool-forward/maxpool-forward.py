import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    def _downsize(value, p=pool_size, s=stride):
        return (value-p)//s+1
        
    H_out, W_out = _downsize(len(X)), _downsize(len(X[0]))
    X_out = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            max = -1000000
            for a in range(pool_size):
                for b in range(pool_size):
                    if X[i*stride+a][j*stride+b] > max:
                        max = X[i*stride+a][j*stride+b]
            X_out[i][j]=max

    return X_out.tolist()