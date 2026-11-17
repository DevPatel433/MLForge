"""Logistic regression, softmax regression and the Perceptron."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..optimization import minimize_batches
from ..utils.math_utils import sigmoid, softmax
from .losses import logistic_loss_grad, softmax_loss_grad


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """Binary logistic regression.

    Parameters
    ----------
    C : float | None
        Inverse L2 regularisation strength (as in scikit-learn); ``None``
        disables the penalty.
    solver : {'gd', 'newton'}
        ``'gd'`` uses the pluggable optimizers / mini-batches; ``'newton'`` is
        iteratively re-weighted least squares (exact Newton steps).
    """

    def __init__(self, learning_rate=0.1, n_iters=1000, C=None, solver="gd",
                 optimizer=None, batch_size=None, tol=0.0, threshold=0.5,
                 fit_intercept=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.C = C
        self.solver = solver
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.tol = tol
        self.threshold = threshold
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    weights = property(lambda self: self.coef_)
    bias = property(lambda self: self.intercept_)
    cost_history = property(lambda self: self.cost_history_)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"LogisticRegression is binary but y has {len(self.classes_)} classes. "
                "Use SoftmaxRegression for multiclass problems.")
        y01 = (y == self.classes_[1]).astype(float)
        n, d = X.shape

        if self.solver == "gd":
            theta, hist = minimize_batches(
                lambda th, idx: logistic_loss_grad(
                    th, X if idx is None else X[idx], y01 if idx is None else y01[idx],
                    self.C, self.fit_intercept),
                np.zeros(d + 1), self.optimizer, self.n_iters, n, self.batch_size,
                self.tol, self.random_state, learning_rate=self.learning_rate)
        elif self.solver == "newton":
            theta, hist = self._newton(X, y01)
        else:
            raise ValueError("solver must be 'gd' or 'newton'")
        self.coef_, self.intercept_ = theta[:-1], float(theta[-1])
        self.cost_history_ = hist
        self.n_features_in_ = d
        return self

    def _newton(self, X, y01):
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))]) if self.fit_intercept else np.hstack([X, np.zeros((n, 1))])
        theta = np.zeros(d + 1)
        reg = np.zeros(d + 1)
        if self.C is not None:
            reg[:d] = 1.0 / (self.C * n)
        hist = []
        for _ in range(min(self.n_iters, 100)):
            loss, grad = logistic_loss_grad(theta, X, y01, self.C, self.fit_intercept)
            hist.append(float(loss))
            p = sigmoid(Xb @ theta)
            W = np.maximum(p * (1 - p), 1e-10)
            H = (Xb * W[:, None]).T @ Xb / n + np.diag(reg) + 1e-10 * np.eye(d + 1)
            if not self.fit_intercept:
                H[-1, -1] = 1.0
            step = np.linalg.solve(H, grad)
            theta = theta - step
            if np.max(np.abs(step)) < 1e-10:
                break
        hist.append(float(logistic_loss_grad(theta, X, y01, self.C, self.fit_intercept)[0]))
        return theta, hist

    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        p1 = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        idx = (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
        return self.classes_[idx]


class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    """Multinomial logistic regression (softmax + cross-entropy)."""

    def __init__(self, learning_rate=0.1, n_iters=1000, C=None, optimizer=None,
                 batch_size=None, tol=0.0, fit_intercept=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.C = C
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    weights = property(lambda self: self.coef_)
    bias = property(lambda self: self.intercept_)
    cost_history = property(lambda self: self.cost_history_)

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        Y = np.zeros((n, k))
        Y[np.arange(n), y_idx] = 1
        theta, hist = minimize_batches(
            lambda th, idx: softmax_loss_grad(
                th, X if idx is None else X[idx], Y if idx is None else Y[idx],
                self.C, self.fit_intercept),
            np.zeros(d * k + k), self.optimizer, self.n_iters, n, self.batch_size,
            self.tol, self.random_state, learning_rate=self.learning_rate)
        self.coef_ = theta[: d * k].reshape(d, k)
        self.intercept_ = theta[d * k:]
        self.cost_history_ = hist
        self.n_features_in_ = d
        return self

    def decision_function(self, X):
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class Perceptron(BaseEstimator, ClassifierMixin):
    """Rosenblatt's perceptron (one-vs-rest for more than two classes).

    Update rule on a mistake: ``w <- w + lr * y_i * x_i`` with ``y_i in {-1, +1}``.
    Converges in a finite number of updates when the data are linearly separable.
    """

    def __init__(self, learning_rate=1.0, n_iters=100, shuffle=True, random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        n_models = 1 if k == 2 else k
        self.coef_ = np.zeros((n_models, d))
        self.intercept_ = np.zeros(n_models)
        rng = np.random.RandomState(self.random_state)
        self.errors_ = []
        for m in range(n_models):
            target = self.classes_[1] if k == 2 else self.classes_[m]
            t = np.where(y == target, 1.0, -1.0)
            w, b = np.zeros(d), 0.0
            for _ in range(self.n_iters):
                order = rng.permutation(n) if self.shuffle else np.arange(n)
                mistakes = 0
                for i in order:
                    if t[i] * (X[i] @ w + b) <= 0:
                        w += self.learning_rate * t[i] * X[i]
                        b += self.learning_rate * t[i]
                        mistakes += 1
                if m == 0:
                    self.errors_.append(mistakes)
                if mistakes == 0:
                    break
            self.coef_[m], self.intercept_[m] = w, b
        self.n_features_in_ = d
        return self

    def decision_function(self, X):
        scores = X @ self.coef_.T + self.intercept_
        return scores.ravel() if scores.shape[1] == 1 else scores

    def predict(self, X):
        s = self.decision_function(X)
        if s.ndim == 1:
            return self.classes_[(s > 0).astype(int)]
        return self.classes_[np.argmax(s, axis=1)]
