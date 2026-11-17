"""Gaussian mixture model fitted with the EM algorithm."""
import numpy as np

from ..base import BaseEstimator
from .kmeans import KMeans


def _log_gauss(X, mean, cov):
    d = X.shape[1]
    L = np.linalg.cholesky(cov)
    diff = np.linalg.solve(L, (X - mean).T)
    return -0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + (diff ** 2).sum(0))


class GaussianMixture(BaseEstimator):
    """Mixture of ``n_components`` full-covariance Gaussians (EM)."""

    def __init__(self, n_components=2, max_iter=200, tol=1e-6, reg_covar=1e-6,
                 random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _log_resp(self, X):
        lp = np.column_stack([np.log(self.weights_[k]) + _log_gauss(X, self.means_[k], self.covariances_[k])
                              for k in range(self.n_components)])
        m = lp.max(1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lp - m).sum(1))
        return lp - lse[:, None], lse

    def fit(self, X, y=None):
        n, d = X.shape
        k = self.n_components
        km = KMeans(k, n_init=3, random_state=self.random_state).fit(X)
        self.means_ = km.cluster_centers_.copy()
        self.weights_ = np.array([(km.labels_ == c).mean() for c in range(k)]).clip(1e-6)
        self.covariances_ = np.array([
            np.cov(X[km.labels_ == c].T) + self.reg_covar * np.eye(d) if (km.labels_ == c).sum() > d
            else np.eye(d) for c in range(k)]).reshape(k, d, d)
        prev, self.log_likelihood_history_ = -np.inf, []
        for it in range(self.max_iter):  # noqa: B007 - `it` is read after the loop
            log_resp, lse = self._log_resp(X)                      # E-step
            resp = np.exp(log_resp)
            nk = resp.sum(0) + 1e-12                               # M-step
            self.weights_ = nk / n
            self.means_ = resp.T @ X / nk[:, None]
            for c in range(k):
                diff = X - self.means_[c]
                self.covariances_[c] = (resp[:, c, None] * diff).T @ diff / nk[c] + self.reg_covar * np.eye(d)
            ll = float(lse.mean())
            self.log_likelihood_history_.append(ll)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        self.n_iter_ = it + 1
        self.lower_bound_ = prev if it == 0 else ll
        return self

    def predict_proba(self, X):
        return np.exp(self._log_resp(X)[0])

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def score_samples(self, X):
        return self._log_resp(X)[1]

    def score(self, X, y=None):
        return float(self.score_samples(X).mean())

    def _n_parameters(self, d):
        k = self.n_components
        return k * d + k * d * (d + 1) // 2 + k - 1

    def bic(self, X):
        return -2 * self.score(X) * len(X) + self._n_parameters(X.shape[1]) * np.log(len(X))

    def aic(self, X):
        return -2 * self.score(X) * len(X) + 2 * self._n_parameters(X.shape[1])

    def sample(self, n=100, random_state=None):
        rng = np.random.RandomState(random_state)
        comp = rng.choice(self.n_components, n, p=self.weights_)
        X = np.array([rng.multivariate_normal(self.means_[c], self.covariances_[c]) for c in comp])
        return X, comp
