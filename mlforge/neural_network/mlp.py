"""Scikit-learn style estimators built on :class:`Sequential`."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.math_utils import softmax
from .layers import ACTIVATIONS, CrossEntropyLoss, Dense, Dropout, MSELoss
from .sequential import Sequential


class _BaseMLP(BaseEstimator):
    def __init__(self, hidden_sizes=(64,), activation="relu", learning_rate=0.01,
                 optimizer="adam", n_epochs=200, batch_size=32, l2=0.0, dropout=0.0,
                 init=None, early_stopping=False, validation_fraction=0.1,
                 patience=10, random_state=None, verbose=0):
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.dropout = dropout
        self.init = init
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

    activation_name = property(lambda self: self.activation)
    lr = property(lambda self: self.learning_rate)

    def _build(self, n_in, n_out):
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        init = self.init or ("xavier" if self.activation in ("tanh", "sigmoid") else "he")
        rng = np.random.RandomState(self.random_state)
        sizes = [n_in] + list(self.hidden_sizes) + [n_out]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Dense(sizes[i], sizes[i + 1], init, rng.randint(2 ** 31 - 1)))
            if i < len(sizes) - 2:
                layers.append(ACTIVATIONS[self.activation]())
                if self.dropout > 0:
                    layers.append(Dropout(self.dropout, rng.randint(2 ** 31 - 1)))
        return Sequential(layers)

    def _train(self, X, Y, loss, metric):
        val = None
        if self.early_stopping:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(len(X))
            n_val = max(1, int(len(X) * self.validation_fraction))
            vi, ti = perm[:n_val], perm[n_val:]
            val, X, Y = (X[vi], Y[vi]), X[ti], Y[ti]
        self.history_ = self.model_.fit(
            X, Y, loss, self.optimizer, self.learning_rate, self.n_epochs, self.batch_size,
            self.l2, val, self.patience if self.early_stopping else None, metric,
            True, self.random_state, self.verbose)
        self.n_epochs_run_ = len(self.history_["loss"])
        self.n_features_in_ = X.shape[1]

    @property
    def loss_history(self):
        return self.history_["loss"]

    @property
    def layers(self):
        return self.model_.layers


class MLPClassifier(_BaseMLP, ClassifierMixin):
    """Multi-layer perceptron classifier (softmax output, cross-entropy loss)."""

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.model_ = self._build(X.shape[1], len(self.classes_))
        acc = lambda out, Y: float(np.mean(out.argmax(1) == Y))
        self._train(X, y_idx, CrossEntropyLoss(), acc)
        return self

    def decision_function(self, X):
        return self.model_.forward(X, training=False)

    def predict_proba(self, X):
        return softmax(self.decision_function(X), axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class MLPRegressor(_BaseMLP, RegressorMixin):
    """MLP regressor with MSE loss; the target is standardised internally."""

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() or 1.0
        self.model_ = self._build(X.shape[1], 1)
        yz = ((y - self.y_mean_) / self.y_std_).reshape(-1, 1)
        self._train(X, yz, MSELoss(), None)
        return self

    def predict(self, X):
        return self.model_.forward(X, training=False).ravel() * self.y_std_ + self.y_mean_
