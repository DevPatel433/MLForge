"""Gradient boosting machines (Friedman, 2001) with regression-tree weak learners."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.math_utils import sigmoid, softmax
from .decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """Least-squares boosting: each tree fits the residuals (negative gradient
    of the squared loss) of the current ensemble, shrunk by ``learning_rate``."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        rng = np.random.RandomState(self.random_state)
        self.init_ = float(y.mean())
        pred = np.full(len(y), self.init_)
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            resid = y - pred
            idx = np.arange(len(y)) if self.subsample >= 1 else rng.choice(
                len(y), max(2, int(self.subsample * len(y))), replace=False)
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf,
                                         random_state=rng.randint(2 ** 31 - 1))
            tree.fit(X[idx], resid[idx])
            pred += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
            self.train_loss_.append(float(np.mean((y - pred) ** 2)))
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        return self

    def staged_predict(self, X):
        pred = np.full(len(X), self.init_)
        for t in self.estimators_:
            pred = pred + self.learning_rate * t.predict(X)
            yield pred.copy()

    def predict(self, X):
        pred = np.full(len(X), self.init_)
        for t in self.estimators_:
            pred += self.learning_rate * t.predict(X)
        return pred


class GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    """Boosting with the (multinomial) deviance. Trees fit the negative
    gradient ``y - p``; leaf values use one Newton step ``sum(r) / sum(p(1-p))``."""

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state

    def _newton_leaves(self, tree, X, resid, hess, k_scale):
        leaves = tree._leaf_for(X)
        groups = {}
        for i, nd in enumerate(leaves):
            groups.setdefault(id(nd), (nd, []))[1].append(i)
        for nd, idx in groups.values():
            idx = np.array(idx)
            nd.value = float(k_scale * resid[idx].sum() / max(hess[idx].sum(), 1e-12))

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, k = len(y_idx), len(self.classes_)
        rng = np.random.RandomState(self.random_state)
        Y = np.eye(k)[y_idx]
        if k == 2:
            p0 = np.clip(Y[:, 1].mean(), 1e-6, 1 - 1e-6)
            self.init_ = np.array([np.log(p0 / (1 - p0))])
            F = np.full((n, 1), self.init_[0])
        else:
            prior = np.clip(Y.mean(0), 1e-6, None)
            self.init_ = np.log(prior)
            F = np.tile(self.init_, (n, 1))
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            P = sigmoid(F) if k == 2 else softmax(F, axis=1)
            trees = []
            cols = 1 if k == 2 else k
            for c in range(cols):
                target = Y[:, 1] if k == 2 else Y[:, c]
                pc = P[:, 0] if k == 2 else P[:, c]
                resid = target - pc
                hess = pc * (1 - pc)
                idx = np.arange(n) if self.subsample >= 1 else rng.choice(
                    n, max(2, int(self.subsample * n)), replace=False)
                tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                             min_samples_leaf=self.min_samples_leaf,
                                             random_state=rng.randint(2 ** 31 - 1))
                tree.fit(X[idx], resid[idx])
                scale = 1.0 if k == 2 else (k - 1) / k
                self._newton_leaves(tree, X[idx], resid[idx], hess[idx], scale)
                F[:, c] += self.learning_rate * tree.predict(X)
                trees.append(tree)
            self.estimators_.append(trees)
            P = sigmoid(F) if k == 2 else softmax(F, axis=1)
            if k == 2:
                self.train_loss_.append(float(-np.mean(Y[:, 1] * np.log(P[:, 0] + 1e-12) + (1 - Y[:, 1]) * np.log(1 - P[:, 0] + 1e-12))))
            else:
                self.train_loss_.append(float(-np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))))
        self.feature_importances_ = np.mean(
            [t.feature_importances_ for trees in self.estimators_ for t in trees], axis=0)
        return self

    def decision_function(self, X):
        k = len(self.classes_)
        F = np.tile(self.init_, (len(X), 1)) if k > 2 else np.full((len(X), 1), self.init_[0])
        for trees in self.estimators_:
            for c, t in enumerate(trees):
                F[:, c] += self.learning_rate * t.predict(X)
        return F[:, 0] if k == 2 else F

    def predict_proba(self, X):
        F = self.decision_function(X)
        if len(self.classes_) == 2:
            p = sigmoid(F)
            return np.column_stack([1 - p, p])
        return softmax(F, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
