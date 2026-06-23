def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    classes = set(y_true) | set(y_pred)
    TP = FP = FN = 0

    for cls in classes:
        for t, p in zip(y_true, y_pred):
            if t == cls and p == cls:
                TP += 1
            elif t != cls and p == cls:
                FP += 1
            elif t == cls and p != cls:
                FN += 1

    return 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0