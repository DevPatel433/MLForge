"""Principal Component Analysis (SVD and covariance-eigendecomposition solvers)."""
import numpy as np

from ..base import BaseEstimator
from ..utils.math_utils import covariance_matrix


class PCA(BaseEstimator):
    """PCA.

    n_components : int | float in (0, 1) | None
        A float keeps the smallest number of components explaining at least
        that fraction of the variance.
    solver : {'svd', 'eigh'} - ``'eigh'`` diagonalises the covariance matrix
        (the textbook derivation); ``'svd'`` is numerically more stable.
    whiten : divide projections by sqrt(explained variance).
    """

    def __init__(self, n_components=2, whiten=False, solver="svd"):
        self.n_components = n_components
        self.whiten = whiten
        self.solver = solver

    # v0.1 attribute names
    components = property(lambda self: self.components_.T)
    mean = property(lambda self: self.mean_)
    explained_variance = property(lambda self: self.explained_variance_)
    explained_variance_ratio = property(lambda self: self.explained_variance_ratio_)

    def fit(self, X, y=None):
        n, d = X.shape
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if self.solver == "eigh":
            vals, vecs = np.linalg.eigh(covariance_matrix(X, ddof=1))
            order = np.argsort(vals)[::-1]
            var, comps = np.maximum(vals[order], 0), vecs[:, order].T
        elif self.solver == "svd":
            _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            var, comps = s ** 2 / (n - 1), Vt
        else:
            raise ValueError("solver must be 'svd' or 'eigh'")
        # deterministic sign convention: largest |entry| of each component is positive
        signs = np.sign(comps[np.arange(len(comps)), np.abs(comps).argmax(1)])
        signs[signs == 0] = 1
        comps = comps * signs[:, None]

        total = var.sum()
        ratio = var / total if total > 0 else var
        k = self.n_components
        if k is None:
            k = len(var)
        elif isinstance(k, float) and 0 < k < 1:
            k = int(np.searchsorted(np.cumsum(ratio), k) + 1)
        k = int(min(k, len(var)))
        self.n_components_ = k
        self.components_ = comps[:k]
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = ratio[:k]
        self.singular_values_ = np.sqrt(var[:k] * (n - 1))
        self.n_features_in_ = d
        return self

    def transform(self, X):
        Z = (X - self.mean_) @ self.components_.T
        return Z / np.sqrt(self.explained_variance_) if self.whiten else Z

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=float)
        if self.whiten:
            Z = Z * np.sqrt(self.explained_variance_)
        return Z @ self.components_ + self.mean_

    def reconstruction_error(self, X):
        return float(np.mean((X - self.inverse_transform(self.transform(X))) ** 2))
