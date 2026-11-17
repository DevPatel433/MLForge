"""Regularised linear regression: Ridge (L2), Lasso (L1) and Elastic Net.

Objectives follow the scikit-learn conventions so results are directly
comparable (see ``benchmarks/``):

* Ridge:       ``||y - Xw||^2 + alpha ||w||^2``
* Lasso:       ``1/(2n) ||y - Xw||^2 + alpha ||w||_1``
* ElasticNet:  ``1/(2n) ||y - Xw||^2 + alpha l1_ratio ||w||_1
                 + alpha (1 - l1_ratio)/2 ||w||^2``
"""
import numpy as np

from ..base import RegressorMixin
from .linear_regression import LinearRegression, _LinearModel


class Ridge(LinearRegression):
    """L2-regularised least squares (closed form or gradient descent)."""

    def __init__(self, alpha=1.0, learning_rate=0.01, n_iters=1000, optimizer=None,
                 fit_intercept=True, solver="normal", batch_size=None, tol=0.0,
                 random_state=None):
        super().__init__(learning_rate, n_iters, optimizer, fit_intercept, solver,
                         batch_size, tol, random_state)
        self.alpha = alpha

    def _alpha(self):
        return self.alpha


def _soft_threshold(rho, lam):
    return np.sign(rho) * np.maximum(np.abs(rho) - lam, 0.0)


class ElasticNet(_LinearModel, RegressorMixin):
    """Elastic Net via cyclic coordinate descent (``solver='cd'``) or proximal
    gradient / ISTA (``solver='ista'``). ``l1_ratio=1`` is the Lasso."""

    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6,
                 fit_intercept=True, solver="cd"):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.solver = solver

    def _objective(self, X, y, w):
        n = X.shape[0]
        r = y - X @ w
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        return (r @ r) / (2 * n) + l1 * np.abs(w).sum() + 0.5 * l2 * (w @ w)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.fit_intercept:
            x_mean, y_mean = X.mean(axis=0), y.mean()
            Xc, yc = X - x_mean, y - y_mean
        else:
            Xc, yc, x_mean, y_mean = X, y, np.zeros(d), 0.0

        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        w = np.zeros(d)
        history = []

        if self.solver == "cd":
            col_sq = (Xc ** 2).sum(axis=0) / n
            r = yc.copy()
            for _ in range(self.max_iter):
                max_delta = 0.0
                for j in range(d):
                    if col_sq[j] == 0:
                        continue
                    old = w[j]
                    rho = Xc[:, j] @ r / n + col_sq[j] * old
                    new = _soft_threshold(rho, l1) / (col_sq[j] + l2)
                    if new != old:
                        r -= Xc[:, j] * (new - old)
                        w[j] = new
                        max_delta = max(max_delta, abs(new - old))
                history.append(self._objective(Xc, yc, w))
                if max_delta < self.tol:
                    break
        elif self.solver == "ista":
            L = np.linalg.norm(Xc, 2) ** 2 / n + l2          # Lipschitz constant
            step = 1.0 / max(L, 1e-12)
            for _ in range(self.max_iter):
                grad = -Xc.T @ (yc - Xc @ w) / n + l2 * w
                w_new = _soft_threshold(w - step * grad, step * l1)
                delta = np.max(np.abs(w_new - w))
                w = w_new
                history.append(self._objective(Xc, yc, w))
                if delta < self.tol:
                    break
        else:
            raise ValueError("solver must be 'cd' or 'ista'")

        self.coef_ = w
        self.intercept_ = float(y_mean - x_mean @ w) if self.fit_intercept else 0.0
        self.cost_history_ = history
        self.n_iter_ = len(history)
        self.n_features_in_ = d
        return self

    def predict(self, X):
        return self._decision(X)


class Lasso(ElasticNet):
    """L1-regularised least squares (sparse solutions)."""

    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-6, fit_intercept=True,
                 solver="cd"):
        super().__init__(alpha=alpha, l1_ratio=1.0, max_iter=max_iter, tol=tol,
                         fit_intercept=fit_intercept, solver=solver)

    def get_params(self):
        p = super().get_params()
        p.pop("l1_ratio", None)
        return p
