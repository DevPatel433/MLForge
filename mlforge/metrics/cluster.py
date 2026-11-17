"""Clustering quality metrics."""
import numpy as np

from ..utils.pairwise import pairwise_distances


def silhouette_samples(X, labels):
    """Per-sample silhouette coefficient ``(b - a) / max(a, b)``."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2 or len(classes) >= len(X):
        raise ValueError("Silhouette needs 2 <= n_clusters <= n_samples - 1.")
    D = pairwise_distances(X)
    n = len(X)
    a = np.zeros(n)
    b = np.full(n, np.inf)
    for c in classes:
        mask = labels == c
        size = mask.sum()
        # mean distance from every point to cluster c
        mean_to_c = D[:, mask].sum(axis=1) / size
        own = mask
        a[own] = D[np.ix_(own, own)].sum(axis=1) / max(size - 1, 1)
        other = ~mask
        b[other] = np.minimum(b[other], mean_to_c[other])
    s = (b - a) / np.maximum(a, b)
    s[np.array([np.sum(labels == l) for l in labels]) == 1] = 0.0
    return np.nan_to_num(s)


def silhouette_score(X, labels):
    return float(np.mean(silhouette_samples(X, labels)))


def davies_bouldin_score(X, labels):
    X = np.asarray(X, dtype=float)
    classes = np.unique(labels)
    cents = np.array([X[labels == c].mean(0) for c in classes])
    scat = np.array([np.mean(np.linalg.norm(X[labels == c] - cents[i], axis=1))
                     for i, c in enumerate(classes)])
    dist = pairwise_distances(cents)
    np.fill_diagonal(dist, np.inf)
    ratio = (scat[:, None] + scat[None, :]) / dist
    return float(np.mean(np.max(ratio, axis=1)))


def calinski_harabasz_score(X, labels):
    X = np.asarray(X, dtype=float)
    classes = np.unique(labels)
    n, k = len(X), len(classes)
    mean = X.mean(0)
    extra = intra = 0.0
    for c in classes:
        Xc = X[labels == c]
        mc = Xc.mean(0)
        extra += len(Xc) * np.sum((mc - mean) ** 2)
        intra += np.sum((Xc - mc) ** 2)
    return float(extra * (n - k) / (intra * (k - 1))) if intra > 0 else 1.0


def _contingency(a, b):
    ca, ia = np.unique(a, return_inverse=True)
    cb, ib = np.unique(b, return_inverse=True)
    m = np.zeros((len(ca), len(cb)))
    np.add.at(m, (ia, ib), 1)
    return m


def adjusted_rand_score(labels_true, labels_pred):
    m = _contingency(labels_true, labels_pred)
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = comb(m).sum()
    sum_a = comb(m.sum(1)).sum()
    sum_b = comb(m.sum(0)).sum()
    total = comb(m.sum())
    expected = sum_a * sum_b / total if total else 0
    max_idx = (sum_a + sum_b) / 2
    return float((sum_ij - expected) / (max_idx - expected)) if max_idx != expected else 1.0


def normalized_mutual_info_score(labels_true, labels_pred):
    m = _contingency(labels_true, labels_pred)
    n = m.sum()
    pij = m / n
    pi, pj = pij.sum(1), pij.sum(0)
    nz = pij > 0
    mi = np.sum(pij[nz] * np.log(pij[nz] / (pi[:, None] * pj[None, :])[nz]))
    h = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
    denom = (h(pi) + h(pj)) / 2
    return float(mi / denom) if denom > 0 else 1.0
