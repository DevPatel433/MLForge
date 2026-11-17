"""K-Means clustering (Lloyd's algorithm) with K-Means++ seeding."""
import numpy as np

from ..base import BaseEstimator
from ..utils.pairwise import pairwise_distances


class KMeans(BaseEstimator):
    """K-Means.

    Parameters
    ----------
    init : {'k-means++', 'random'}
    n_init : int - independent restarts; the run with the lowest inertia wins.
    ``history`` stores the centroids after every iteration of the best run
    (used by :func:`mlforge.visualization.animate_kmeans`).
    """

    def __init__(self, n_clusters=3, init="k-means++", n_init=5, max_iters=300,
                 tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    max_iter = property(lambda self: self.max_iters)
    centroids = property(lambda self: self.cluster_centers_)

    def _seed(self, X, rng):
        n = len(X)
        if self.init == "random":
            return X[rng.choice(n, self.n_clusters, replace=False)].copy()
        centers = [X[rng.randint(n)]]
        d2 = ((X - centers[0]) ** 2).sum(1)
        for _ in range(1, self.n_clusters):
            total = d2.sum()
            probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
            centers.append(X[rng.choice(n, p=probs)])
            d2 = np.minimum(d2, ((X - centers[-1]) ** 2).sum(1))
        return np.array(centers)

    def _lloyd(self, X, centers):
        history = [centers.copy()]
        for it in range(self.max_iters):  # noqa: B007 - `it` is read after the loop
            D = pairwise_distances(X, centers, "sqeuclidean")
            labels = D.argmin(1)
            new = centers.copy()
            for k in range(self.n_clusters):
                members = X[labels == k]
                if len(members):
                    new[k] = members.mean(0)
                else:                       # re-seed an empty cluster at the worst-fit point
                    new[k] = X[D.min(1).argmax()]
            shift = np.sqrt(((new - centers) ** 2).sum())
            centers = new
            history.append(centers.copy())
            if shift < self.tol:
                break
        D = pairwise_distances(X, centers, "sqeuclidean")
        labels = D.argmin(1)
        return centers, labels, float(D[np.arange(len(X)), labels].sum()), it + 1, history

    def fit(self, X, y=None):
        if self.n_clusters > len(X):
            raise ValueError(f"n_clusters={self.n_clusters} exceeds n_samples={len(X)}.")
        rng = np.random.RandomState(self.random_state)
        best = None
        for _ in range(max(1, self.n_init)):
            run = self._lloyd(X, self._seed(X, rng))
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_, self.history = best
        return self

    def predict(self, X):
        return pairwise_distances(X, self.cluster_centers_, "sqeuclidean").argmin(1)

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def transform(self, X):
        return pairwise_distances(X, self.cluster_centers_)


def elbow_curve(X, k_range=range(1, 11), random_state=0, **kwargs):
    """Inertia for each ``k`` (the *elbow method*)."""
    return np.array([KMeans(k, random_state=random_state, **kwargs).fit(X).inertia_ for k in k_range])


def silhouette_curve(X, k_range=range(2, 11), random_state=0, **kwargs):
    from ..metrics.cluster import silhouette_score
    return np.array([silhouette_score(X, KMeans(k, random_state=random_state, **kwargs).fit(X).labels_)
                     for k in k_range])
