import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    M, N = len(A),len(A[0])
    A_t = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            A_t[i, j] = A[j][i]
    return A_t
