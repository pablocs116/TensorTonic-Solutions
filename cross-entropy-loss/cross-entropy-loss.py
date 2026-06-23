import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    loss = 0
    N = len(y_true)
    for y_i,pred_i in zip(y_true, y_pred):
        prob = pred_i[y_i]
        loss+=np.log(prob)

    return -loss/N

    