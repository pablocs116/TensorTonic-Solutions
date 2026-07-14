import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    ss_tot = np.sum([(y-np.mean(y_true))**2 for y in y_true])
    if y_true == y_pred:
        return 1.0
    elif ss_tot == 0:
        return 0.0
    else:
        return 1 - (np.sum([(y-y_t)**2 for y,y_t in zip(y_pred, y_true)]))/ss_tot