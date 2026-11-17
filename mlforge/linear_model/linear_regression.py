"""Ordinary least squares and polynomial regression."""
import numpy as np

from ..base import BaseEstimator, RegressorMixin
from ..optimization import minimize_batches
from .losses import mse_loss_grad


class _LinearModel(BaseEstimator):
    """Common attribute aliases (``weights`` / ``bias`` / ``cost_history``)."""

    @property
    def weights(self):
        return self.coef_

    @property
    def bias(self):
        return self.intercept_

    @property
    def cost_history(self):
        return self.cost_history_

    def _decision(self, X):
        return X @ self.coef_ + self.intercept_


class LinearRegression(_LinearModel, RegressorMixin):
    """Ordinary least squares.

    Parameters
    ----------
    learning_rate, n_iters, optimizer, batch_size, tol, random_state :
        Gradient-descent settings (used when ``solver != 'normal'``).
        ``optimizer`` may be a name (``'adam'``), class or instance;
        ``batch_size=None`` is batch GD, ``1`` is stochastic GD.
    solver : {'normal', 'gd'}
        ``'normal'`` solves the least-squares problem in closed form
        (via ``numpy.linalg.lstsq``); ``'gd'`` uses gradient descent.
    fit_intercept : bool
    """

    def __init__(self, learning_rate=0.01, n_iters=1000, optimizer=None,
                 fit_intercept=True, solver="normal", batch_size=None, tol=0.0,
                 random_state=None):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.optimizer = optimizer
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.batch_size = batch_size
        self.tol = tol
        self.random_state = random_state

    def _alpha(self):
        return 0.0

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        if self.solver == "normal":
            self.coef_, self.intercept_ = _solve_ridge(X, y, self._alpha(), self.fit_intercept)
            self.cost_history_ = [float(np.mean((self._decision(X) - y) ** 2))]
        elif self.solver == "gd":
            theta, hist = minimize_batches(
                lambda th, idx: mse_loss_grad(
                    th, X if idx is None else X[idx], y if idx is None else y[idx],
                    self._alpha(), self.fit_intercept),
                np.zeros(d + 1), self.optimizer, self.n_iters, n, self.batch_size,
                self.tol, self.random_state, learning_rate=self.learning_rate)
            self.coef_, self.intercept_ = theta[:-1], float(theta[-1])
            self.cost_history_ = hist
        else:
            raise ValueError("solver must be 'normal' or 'gd'")
        self.n_features_in_ = d
        return self

    def predict(self, X):
        return self._decision(X)


def _solve_ridge(X, y, alpha, fit_intercept):
    """Closed-form (ridge) least squares on centred data. ``alpha=0`` is OLS."""
    if fit_intercept:
        x_mean, y_mean = X.mean(axis=0), y.mean()
        Xc, yc = X - x_mean, y - y_mean
    else:
        Xc, yc, x_mean, y_mean = X, y, 0.0, 0.0
    if alpha == 0:
        w = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    else:
        A = Xc.T @ Xc + alpha * np.eye(Xc.shape[1])
        w = np.linalg.solve(A, Xc.T @ yc)
    b = float(y_mean - x_mean @ w) if fit_intercept else 0.0
    return w, b


class PolynomialRegression(LinearRegression):
    """Linear regression on polynomial (and interaction) features.

    The expanded features are standardised internally so that gradient
    descent stays well conditioned regardless of ``degree``.
    """

    _allow_1d = True

    def __init__(self, degree=2, interaction_only=False, learning_rate=0.05,
                 n_iters=2000, optimizer=None, fit_intercept=True, solver="normal",
                 batch_size=None, tol=0.0, random_state=None):
        super().__init__(learning_rate, n_iters, optimizer, fit_intercept, solver,
                         batch_size, tol, random_state)
        self.degree = degree
        self.interaction_only = interaction_only

    def _expand(self, X, fit=False):
        from ..preprocessing import PolynomialFeatures, StandardScaler
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if fit:
            self._poly = PolynomialFeatures(self.degree, include_bias=False,
                                            interaction_only=self.interaction_only)
            Z = self._poly.fit_transform(X)
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(Z)
        return self._scaler.transform(self._poly.transform(X))

    def fit(self, X, y):
        return LinearRegression.fit.__wrapped__(self, self._expand(X, fit=True), y)

    def predict(self, X):
        return self._decision(self._expand(X))
