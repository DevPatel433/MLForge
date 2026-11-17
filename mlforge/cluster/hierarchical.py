"""Agglomerative hierarchical clustering and DBSCAN."""
import numpy as np

from ..base import BaseEstimator
from ..utils.pairwise import pairwise_distances


def linkage(X, method="average"):
    """Bottom-up clustering. Returns a SciPy-compatible linkage matrix with
    rows ``[idx_a, idx_b, distance, size]`` (Lance-Williams updates).

    The closest pair is found from cached per-row minima, which are repaired
    only for rows whose nearest neighbour was just merged (≈ O(n^2) overall
    instead of rescanning the whole matrix at every step).
    """
    if method not in ("single", "complete", "average", "ward"):
        raise ValueError("method must be single, complete, average or ward")
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = pairwise_distances(X, metric="sqeuclidean" if method == "ward" else "euclidean")
    np.fill_diagonal(D, np.inf)
    sizes = np.ones(n)
    ids = np.arange(n)                           # slot -> cluster id
    active = np.ones(n, dtype=bool)
    rowarg = D.argmin(axis=1)
    rowmin = D[np.arange(n), rowarg]
    Z = np.empty((n - 1, 4))
    for step in range(n - 1):
        i = int(np.argmin(rowmin))
        j = int(rowarg[i])
        if i > j:
            i, j = j, i
        dij = D[i, j]
        na, nb = sizes[i], sizes[j]
        Z[step] = (ids[i], ids[j], np.sqrt(dij) if method == "ward" else dij, na + nb)
        if method == "single":
            new = np.minimum(D[i], D[j])
        elif method == "complete":
            new = np.maximum(D[i], D[j])
        elif method == "average":
            new = (na * D[i] + nb * D[j]) / (na + nb)
        else:                                    # ward
            new = ((na + sizes) * D[i] + (nb + sizes) * D[j] - sizes * dij) / (na + nb + sizes)
        new[~active] = np.inf
        new[i] = new[j] = np.inf
        D[i, :] = new
        D[:, i] = new
        D[j, :] = np.inf
        D[:, j] = np.inf
        active[j] = False
        sizes[i] = na + nb
        ids[i] = n + step
        rowmin[j] = np.inf
        rowarg[j] = j
        rowmin[i], rowarg[i] = new.min(), int(new.argmin())
        # repair cached minima that pointed at a merged cluster, or that i now beats
        stale = np.where(active & ((rowarg == i) | (rowarg == j)) & (np.arange(n) != i))[0]
        if len(stale):
            rowarg[stale] = D[stale].argmin(axis=1)
            rowmin[stale] = D[stale, rowarg[stale]]
        better = active & (D[:, i] < rowmin) & (np.arange(n) != i)
        rowmin[better] = D[better, i]
        rowarg[better] = i
    return Z


def fcluster(Z, n_clusters):
    """Cut a linkage matrix into ``n_clusters`` flat clusters."""
    n = len(Z) + 1
    parent = list(range(2 * n - 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for step in range(n - n_clusters):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        parent[find(a)] = n + step
        parent[find(b)] = n + step
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


class AgglomerativeClustering(BaseEstimator):
    def __init__(self, n_clusters=2, linkage="ward"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X, y=None):
        self.linkage_matrix_ = linkage(X, self.linkage)
        self.labels_ = fcluster(self.linkage_matrix_, self.n_clusters)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


class DBSCAN(BaseEstimator):
    """Density-based clustering; noise points get label ``-1``."""

    def __init__(self, eps=0.5, min_samples=5, metric="euclidean"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X, y=None):
        D = pairwise_distances(X, metric=self.metric)
        neigh = [np.where(D[i] <= self.eps)[0] for i in range(len(X))]
        core = np.array([len(nb) >= self.min_samples for nb in neigh])
        labels = np.full(len(X), -1)
        cid = 0
        for i in range(len(X)):
            if labels[i] != -1 or not core[i]:
                continue
            labels[i] = cid
            queue = [i]
            while queue:
                p = queue.pop()
                if not core[p]:
                    continue
                for q in neigh[p]:
                    if labels[q] == -1:
                        labels[q] = cid
                        queue.append(q)
            cid += 1
        self.labels_ = labels
        self.core_sample_indices_ = np.where(core)[0]
        self.n_clusters_ = cid
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
