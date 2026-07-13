import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    if average == "binary":
        tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
        fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
        fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        return {"accuracy": accuracy, "precision": float(precision),
                "recall": float(recall), "f1": float(f1)}

    classes = np.unique(np.concatenate([y_true, y_pred]))
    tp = np.zeros(len(classes))
    fp = np.zeros(len(classes))
    fn = np.zeros(len(classes))
    support = np.zeros(len(classes))

    for i, c in enumerate(classes):
        tp[i] = np.sum((y_pred == c) & (y_true == c))
        fp[i] = np.sum((y_pred == c) & (y_true != c))
        fn[i] = np.sum((y_pred != c) & (y_true == c))
        support[i] = np.sum(y_true == c)

    precision_pc = np.array([safe_div(tp[i], tp[i] + fp[i]) for i in range(len(classes))])
    recall_pc = np.array([safe_div(tp[i], tp[i] + fn[i]) for i in range(len(classes))])
    f1_pc = np.array([safe_div(2 * precision_pc[i] * recall_pc[i], precision_pc[i] + recall_pc[i])
                       for i in range(len(classes))])

    if average == "micro":
        tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
        precision = safe_div(tp_s, tp_s + fp_s)
        recall = safe_div(tp_s, tp_s + fn_s)
        f1 = safe_div(2 * precision * recall, precision + recall)

    elif average == "macro":
        precision, recall, f1 = precision_pc.mean(), recall_pc.mean(), f1_pc.mean()

    elif average == "weighted":
        total = support.sum()
        precision = safe_div(np.sum(precision_pc * support), total)
        recall = safe_div(np.sum(recall_pc * support), total)
        f1 = safe_div(np.sum(f1_pc * support), total)

    else:
        raise ValueError(f"Unknown average: {average}")

    return {"accuracy": accuracy, "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}