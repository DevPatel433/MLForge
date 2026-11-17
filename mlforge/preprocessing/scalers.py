"""Feature scaling transformers."""
import numpy as np

from ..base import BaseEstimator


class StandardScaler(BaseEstimator):
    """Remove the mean and scale to unit variance (population std, as sklearn)."""

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, X, y=None):
        mean = X.mean(axis=0)
        self.mean_ = mean if self.with_mean else np.zeros(X.shape[1])
        scale = X.std(axis=0) if self.with_std else np.ones(X.shape[1])
        # A constant column has std == 0 mathematically, but rounding in the mean makes it
        # ~eps*|mean| in floating point; dividing that noise by itself would yield +-1 garbage.
        constant = scale <= 10 * np.finfo(float).eps * np.abs(mean)
        scale[constant | (scale == 0)] = 1.0
        self.scale_ = scale
        self.var_ = X.var(axis=0)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator):
    """Scale each feature to ``feature_range`` (default [0, 1])."""

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        rng = self.data_max_ - self.data_min_
        rng[rng == 0] = 1.0
        self.data_range_ = rng
        return self

    def transform(self, X):
        lo, hi = self.feature_range
        return (X - self.data_min_) / self.data_range_ * (hi - lo) + lo

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        lo, hi = self.feature_range
        return (X - lo) / (hi - lo) * self.data_range_ + self.data_min_

    # backwards-compatible names from v0.1
    min_ = property(lambda self: self.data_min_)
    max_ = property(lambda self: self.data_max_)
    range_ = property(lambda self: self.data_range_)


class MaxAbsScaler(BaseEstimator):
    def fit(self, X, y=None):
        s = np.abs(X).max(axis=0)
        s[s == 0] = 1.0
        self.scale_ = s
        return self

    def transform(self, X):
        return X / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_


class RobustScaler(BaseEstimator):
    """Scale with the median and inter-quartile range (robust to outliers)."""

    def __init__(self, quantile_range=(25.0, 75.0)):
        self.quantile_range = quantile_range

    def fit(self, X, y=None):
        self.center_ = np.median(X, axis=0)
        lo, hi = np.percentile(X, self.quantile_range, axis=0)
        iqr = hi - lo
        iqr[iqr == 0] = 1.0
        self.scale_ = iqr
        return self

    def transform(self, X):
        return (X - self.center_) / self.scale_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return X * self.scale_ + self.center_


class Normalizer(BaseEstimator):
    """Rescale each *sample* to unit norm (``'l1'``, ``'l2'`` or ``'max'``)."""

    def __init__(self, norm="l2"):
        self.norm = norm

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.norm == "l2":
            n = np.linalg.norm(X, axis=1, keepdims=True)
        elif self.norm == "l1":
            n = np.abs(X).sum(axis=1, keepdims=True)
        elif self.norm == "max":
            n = np.abs(X).max(axis=1, keepdims=True)
        else:
            raise ValueError("norm must be 'l1', 'l2' or 'max'")
        n[n == 0] = 1.0
        return X / n

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
