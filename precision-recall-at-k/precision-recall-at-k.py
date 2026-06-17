def precision_recall_at_k(recommended, relevant, k):
    """Compute precision@k and recall@k."""
    hits = set(recommended[:k]) & set(relevant)
    n_hits = len(hits)
    precision = n_hits / k if k > 0 else 0.0
    recall = n_hits / len(relevant) if relevant else 0.0
    return [precision, recall]