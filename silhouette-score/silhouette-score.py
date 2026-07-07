import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)

    # Pairwise distance matrix (n_samples, n_samples)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    s = np.zeros(n_samples)

    for i in range(n_samples):
        own_label = labels[i]
        own_mask = (labels == own_label)
        own_mask[i] = False  # exclude self

        if np.sum(own_mask) == 0:
            # singleton cluster -> silhouette defined as 0
            s[i] = 0.0
            continue

        a_i = np.mean(dist[i, own_mask])

        b_i = np.inf
        for other_label in unique_labels:
            if other_label == own_label:
                continue
            other_mask = (labels == other_label)
            mean_dist_other = np.mean(dist[i, other_mask])
            b_i = min(b_i, mean_dist_other)

        s[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(s)