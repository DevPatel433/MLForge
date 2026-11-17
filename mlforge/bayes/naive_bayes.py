"""Naive Bayes classifiers: Gaussian, Multinomial and Bernoulli."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin


class _NB(BaseEstimator, ClassifierMixin):
    def predict_proba(self, X):
        jll = self._joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        m = jll.max(axis=1, keepdims=True)
        return jll - (m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True)))

    def predict(self, X):
        return self.classes_[np.argmax(self._joint_log_likelihood(X), axis=1)]

    def _priors(self, y_idx, k):
        counts = np.bincount(y_idx, minlength=k).astype(float)
        return np.log(counts / counts.sum()) if self.priors is None else np.log(np.asarray(self.priors))


class GaussianNB(_NB):
    """Gaussian NB: ``P(x_j | c) = N(mu_cj, var_cj)`` with variance smoothing."""

    def __init__(self, var_smoothing=1e-9, priors=None):
        self.var_smoothing = var_smoothing
        self.priors = priors

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k, d = len(self.classes_), X.shape[1]
        self.theta_, self.var_ = np.zeros((k, d)), np.zeros((k, d))
        for c in range(k):
            Xc = X[y_idx == c]
            self.theta_[c], self.var_[c] = Xc.mean(0), Xc.var(0)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()
        self.var_ += self.epsilon_
        self.class_log_prior_ = self._priors(y_idx, k)
        self.class_prior_ = np.exp(self.class_log_prior_)
        return self

    def _joint_log_likelihood(self, X):
        out = []
        for c in range(len(self.classes_)):
            ll = -0.5 * np.sum(np.log(2 * np.pi * self.var_[c]))
            ll = ll - 0.5 * np.sum((X - self.theta_[c]) ** 2 / self.var_[c], axis=1)
            out.append(self.class_log_prior_[c] + ll)
        return np.column_stack(out)


class MultinomialNB(_NB):
    """Multinomial NB for count features with Laplace/Lidstone smoothing ``alpha``."""

    def __init__(self, alpha=1.0, priors=None):
        self.alpha = alpha
        self.priors = priors

    def fit(self, X, y):
        if np.any(X < 0):
            raise ValueError("MultinomialNB requires non-negative feature values.")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        counts = np.array([X[y_idx == c].sum(0) for c in range(k)]) + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(1, keepdims=True))
        self.class_log_prior_ = self._priors(y_idx, k)
        return self

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_


class BernoulliNB(_NB):
    """Bernoulli NB for binary features (``binarize`` thresholds the input)."""

    def __init__(self, alpha=1.0, binarize=0.0, priors=None):
        self.alpha = alpha
        self.binarize = binarize
        self.priors = priors

    def fit(self, X, y):
        Xb = (X > self.binarize).astype(float) if self.binarize is not None else X
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        n_c = np.array([(y_idx == c).sum() for c in range(k)], dtype=float)
        feat = np.array([Xb[y_idx == c].sum(0) for c in range(k)]) + self.alpha
        self.feature_prob_ = feat / (n_c[:, None] + 2 * self.alpha)
        self.class_log_prior_ = self._priors(y_idx, k)
        return self

    def _joint_log_likelihood(self, X):
        Xb = (X > self.binarize).astype(float) if self.binarize is not None else X
        lp, ln = np.log(self.feature_prob_), np.log(1 - self.feature_prob_)
        return Xb @ (lp - ln).T + ln.sum(1) + self.class_log_prior_
