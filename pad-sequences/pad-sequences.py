import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    if N == 0:
        L = max_len or 0
        return np.zeros((0, L), dtype=float)

    L = max_len if max_len is not None else max(len(seq) for seq in seqs)
    seqs = [list(seq)[:L] for seq in seqs]
    x = [list(seq) + [pad_value] * (L - len(seq)) for seq in seqs]
    return np.array(x)