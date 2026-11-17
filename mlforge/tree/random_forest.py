"""Random forests: bagged CART trees with per-split feature subsampling."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


class _BaseForest(BaseEstimator):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", bootstrap=True,
                 oob_score=False, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state

    def _make_tree(self, seed):
        raise NotImplementedError

    def _fit_forest(self, X, y):
        n = X.shape[0]
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self._inbag = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, n, n) if self.bootstrap else np.arange(n)
            tree = self._make_tree(rng.randint(2 ** 31 - 1))
            self._fit_tree(tree, X[idx], y[idx])
            self.estimators_.append(tree)
            self._inbag.append(np.bincount(idx, minlength=n) > 0)
        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        self.n_features_in_ = X.shape[1]


class RandomForestClassifier(_BaseForest, ClassifierMixin):
    """Bagging + random feature subsets + soft voting. ``oob_score=True``
    computes an out-of-bag accuracy estimate (``oob_score_``)."""

    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features="sqrt", criterion="gini", min_samples_leaf=1,
                 bootstrap=True, oob_score=False, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                         max_features, bootstrap, oob_score, random_state)
        self.criterion = criterion

    def _make_tree(self, seed):
        return DecisionTreeClassifier(
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, criterion=self.criterion,
            max_features=self.max_features, random_state=seed)

    def _fit_tree(self, tree, X, y):
        tree.classes_ = self.classes_
        tree._n_classes = len(self.classes_)
        tree._grow(X, y)
        tree._is_fitted = True

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._fit_forest(X, y_idx)
        if self.oob_score and self.bootstrap:
            proba = np.zeros((len(y_idx), len(self.classes_)))
            counts = np.zeros(len(y_idx))
            for tree, inbag in zip(self.estimators_, self._inbag):
                oob = ~inbag
                if oob.any():
                    proba[oob] += tree.predict_proba(X[oob])
                    counts[oob] += 1
            seen = counts > 0
            self.oob_score_ = float(np.mean(np.argmax(proba[seen], axis=1) == y_idx[seen]))
        return self

    def predict_proba(self, X):
        return np.mean([t.predict_proba(X) for t in self.estimators_], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class RandomForestRegressor(_BaseForest, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 max_features=1.0, min_samples_leaf=1, bootstrap=True,
                 oob_score=False, random_state=None):
        super().__init__(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                         max_features, bootstrap, oob_score, random_state)

    def _make_tree(self, seed):
        return DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
            random_state=seed)

    def _fit_tree(self, tree, X, y):
        tree._grow(X, y)
        tree._is_fitted = True

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self._fit_forest(X, y)
        if self.oob_score and self.bootstrap:
            pred, counts = np.zeros(len(y)), np.zeros(len(y))
            for tree, inbag in zip(self.estimators_, self._inbag):
                oob = ~inbag
                if oob.any():
                    pred[oob] += tree.predict(X[oob])
                    counts[oob] += 1
            seen = counts > 0
            from ..metrics.regression import r2_score
            self.oob_score_ = r2_score(y[seen], pred[seen] / counts[seen])
        return self

    def predict(self, X):
        return np.mean([t.predict(X) for t in self.estimators_], axis=0)
