import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    H_out ,W_out = x.shape[2] - W.shape[2] + 1, x.shape[3] - W.shape[3] + 1
    y = np.zeros((x.shape[0], W.shape[0], H_out, W_out))
    for n in range(x.shape[0]):
        for c_out in range(W.shape[0]):
            for i in range(H_out):
                for j in range(W_out):
                    for c_i in range(x.shape[1]):
                        for u in range(W.shape[2]):
                            for v in range(W.shape[3]):
                                y[n][c_out][i][j] += x[n][c_i][u+i][v+j]*W[c_out][c_i][u][v]
                    y[n][c_out][i][j] += b[c_out]

    return y 
                
        