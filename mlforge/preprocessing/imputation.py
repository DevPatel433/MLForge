"""Missing-value handling."""
import numpy as np

from ..base import BaseEstimator


class SimpleImputer(BaseEstimator):
    """Replace NaNs using ``'mean'``, ``'median'``, ``'most_frequent'`` or ``'constant'``."""

    _allow_nan = True

    def __init__(self, strategy="mean", fill_value=0.0, add_indicator=False):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator

    def fit(self, X, y=None):
        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == "median":
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == "most_frequent":
            fill = []
            for j in range(X.shape[1]):
                col = X[~np.isnan(X[:, j]), j]
                vals, counts = np.unique(col, return_counts=True)
                fill.append(vals[np.argmax(counts)] if len(vals) else 0.0)
            self.statistics_ = np.array(fill)
        elif self.strategy == "constant":
            self.statistics_ = np.full(X.shape[1], self.fill_value, dtype=float)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        self.fill_ = self.statistics_
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        mask = np.isnan(X)
        rows, cols = np.where(mask)
        X[rows, cols] = self.statistics_[cols]
        if self.add_indicator:
            keep = mask.any(axis=0)
            return np.hstack([X, mask[:, keep].astype(float)])
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class KNNImputer(BaseEstimator):
    """Impute each missing entry with the mean of that feature among the ``k``
    nearest rows (distance over mutually observed features)."""

    _allow_nan = True

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        self._X = np.array(X, dtype=float)
        self._col_mean = np.nanmean(self._X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        ref = self._X
        ref_mask = np.isnan(ref)
        for i in np.where(np.isnan(X).any(axis=1))[0]:
            obs = ~np.isnan(X[i])
            if not obs.any():
                X[i] = self._col_mean
                continue
            diff = (ref[:, obs] - X[i, obs]) ** 2
            valid = ~ref_mask[:, obs]
            cnt = valid.sum(axis=1)
            dist = np.where(cnt > 0, np.sqrt(np.where(valid, diff, 0).sum(1) * obs.sum() / np.maximum(cnt, 1)), np.inf)
            for j in np.where(np.isnan(X[i]))[0]:
                cand = np.where(~ref_mask[:, j] & np.isfinite(dist))[0]
                if len(cand) == 0:
                    X[i, j] = self._col_mean[j]
                    continue
                nn = cand[np.argsort(dist[cand])[: self.n_neighbors]]
                X[i, j] = ref[nn, j].mean()
        return X

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
