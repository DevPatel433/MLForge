"""K-Nearest Neighbours with vectorised brute-force search."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.pairwise import pairwise_distances


class _KNNBase(BaseEstimator):
    def __init__(self, k=5, metric="euclidean", weights="uniform", p=2, weighted=None):
        self.k = k
        self.metric = metric
        self.weights = "distance" if weighted else weights
        self.p = p
        self.weighted = weighted

    def _store(self, X):
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.k > len(X):
            raise ValueError(f"k={self.k} exceeds the number of training samples ({len(X)}).")
        self.X_train = X

    def kneighbors(self, X, k=None, return_distance=True):
        k = k or self.k
        out_d, out_i = [], []
        step = max(1, int(5e6 // max(1, len(self.X_train))))
        for s in range(0, len(X), step):
            D = pairwise_distances(X[s:s + step], self.X_train, self.metric, self.p)
            # stable sort => ties are broken by training-sample index, so results
            # are fully deterministic (argpartition would pick ties arbitrarily)
            idx = np.argsort(D, axis=1, kind="stable")[:, :k]
            part = np.take_along_axis(D, idx, 1)
            out_i.append(idx)
            out_d.append(part)
        idx, dist = np.vstack(out_i), np.vstack(out_d)
        return (dist, idx) if return_distance else idx

    def _w(self, dist):
        if self.weights == "distance":
            w = 1.0 / np.maximum(dist, 1e-12)
            exact = dist <= 1e-12
            rows = exact.any(axis=1)
            w[rows] = exact[rows].astype(float)      # exact matches dominate
            return w
        return np.ones_like(dist)


class KNeighborsClassifier(_KNNBase, ClassifierMixin):
    def fit(self, X, y):
        self._store(X)
        self.classes_, self._y = np.unique(y, return_inverse=True)
        return self

    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        w = self._w(dist)
        votes = np.zeros((len(X), len(self.classes_)))
        labels = self._y[idx]
        for c in range(len(self.classes_)):
            votes[:, c] = (w * (labels == c)).sum(axis=1)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KNeighborsRegressor(_KNNBase, RegressorMixin):
    def fit(self, X, y):
        self._store(X)
        self._y = np.asarray(y, dtype=float)
        return self

    def predict(self, X):
        dist, idx = self.kneighbors(X)
        w = self._w(dist)
        return (w * self._y[idx]).sum(axis=1) / w.sum(axis=1)
