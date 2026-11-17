"""Cross-validation, hyper-parameter search and diagnostic curves."""
import itertools
import time

import numpy as np

from .. import metrics as M
from ..base import BaseEstimator, ClassifierMixin
from .split import KFold, StratifiedKFold


# ------------------------------------------------------------------- scoring
def _proba_pos(est, X):
    p = est.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p


_SCORERS = {
    "accuracy": lambda e, X, y: M.accuracy_score(y, e.predict(X)),
    "balanced_accuracy": lambda e, X, y: M.balanced_accuracy_score(y, e.predict(X)),
    "f1": lambda e, X, y: M.f1_score(y, e.predict(X)),
    "f1_macro": lambda e, X, y: M.f1_score(y, e.predict(X), average="macro"),
    "precision": lambda e, X, y: M.precision_score(y, e.predict(X)),
    "recall": lambda e, X, y: M.recall_score(y, e.predict(X)),
    "roc_auc": lambda e, X, y: M.roc_auc_score(y, e.predict_proba(X)),
    "neg_log_loss": lambda e, X, y: -M.log_loss(y, e.predict_proba(X)),
    "r2": lambda e, X, y: M.r2_score(y, e.predict(X)),
    "neg_mean_squared_error": lambda e, X, y: -M.mean_squared_error(y, e.predict(X)),
    "neg_root_mean_squared_error": lambda e, X, y: -M.root_mean_squared_error(y, e.predict(X)),
    "neg_mean_absolute_error": lambda e, X, y: -M.mean_absolute_error(y, e.predict(X)),
}


def get_scorer(scoring):
    if scoring is None:
        return lambda e, X, y: e.score(X, y)
    if callable(scoring):
        return scoring
    if scoring not in _SCORERS:
        raise ValueError(f"Unknown scoring {scoring!r}. Choose from {sorted(_SCORERS)}.")
    return _SCORERS[scoring]


def _is_classifier(est):
    if isinstance(est, ClassifierMixin):
        return True
    steps = getattr(est, "steps", None)
    return bool(steps) and _is_classifier(steps[-1][1])


def check_cv(cv, y=None, estimator=None, random_state=None):
    if hasattr(cv, "split"):
        return cv
    if estimator is not None and _is_classifier(estimator) and y is not None:
        return StratifiedKFold(cv, shuffle=True, random_state=random_state)
    return KFold(cv, shuffle=True, random_state=random_state)


# --------------------------------------------------------------- CV utilities
def cross_val_score(estimator, X, y, cv=5, scoring=None, random_state=0):
    X, y = np.asarray(X), np.asarray(y)
    scorer = get_scorer(scoring)
    scores = []
    for tr, te in check_cv(cv, y, estimator, random_state).split(X, y):
        model = estimator.clone().fit(X[tr], y[tr])
        scores.append(scorer(model, X[te], y[te]))
    return np.array(scores)


def cross_val_predict(estimator, X, y, cv=5, random_state=0):
    X, y = np.asarray(X), np.asarray(y)
    out = np.empty(len(y), dtype=y.dtype if not _is_classifier(estimator) else y.dtype)
    out = out.astype(float) if not _is_classifier(estimator) else out
    for tr, te in check_cv(cv, y, estimator, random_state).split(X, y):
        out[te] = estimator.clone().fit(X[tr], y[tr]).predict(X[te])
    return out


class ParameterGrid:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def __iter__(self):
        keys = sorted(self.param_grid)
        for combo in itertools.product(*(self.param_grid[k] for k in keys)):
            yield dict(zip(keys, combo))

    def __len__(self):
        return int(np.prod([len(v) for v in self.param_grid.values()]))


class GridSearchCV(BaseEstimator):
    """Exhaustive search over a parameter grid with cross-validation."""

    _allow_nan = True

    def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True, random_state=0):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.refit = refit
        self.random_state = random_state

    def _candidates(self):
        return list(ParameterGrid(self.param_grid))

    def fit(self, X, y):
        scorer = get_scorer(self.scoring)
        splitter = check_cv(self.cv, y, self.estimator, self.random_state)
        splits = list(splitter.split(X, y))
        results = {"params": [], "mean_test_score": [], "std_test_score": [], "mean_fit_time": []}
        for params in self._candidates():
            scores, t0 = [], time.perf_counter()
            for tr, te in splits:
                model = self.estimator.clone().set_params(**params).fit(X[tr], y[tr])
                scores.append(scorer(model, X[te], y[te]))
            results["params"].append(params)
            results["mean_test_score"].append(float(np.mean(scores)))
            results["std_test_score"].append(float(np.std(scores)))
            results["mean_fit_time"].append((time.perf_counter() - t0) / len(splits))
        results = {k: np.array(v, dtype=object if k == "params" else float) for k, v in results.items()}
        order = np.argsort(-results["mean_test_score"], kind="mergesort")
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(1, len(order) + 1)
        results["rank_test_score"] = rank
        self.cv_results_ = results
        best = int(order[0])
        self.best_index_ = best
        self.best_params_ = results["params"][best]
        self.best_score_ = float(results["mean_test_score"][best])
        if self.refit:
            self.best_estimator_ = self.estimator.clone().set_params(**self.best_params_).fit(X, y)
            if hasattr(self.best_estimator_, "classes_"):
                self.classes_ = self.best_estimator_.classes_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)

    def score(self, X, y):
        return get_scorer(self.scoring)(self.best_estimator_, X, y)


class RandomizedSearchCV(GridSearchCV):
    """Random sample of ``n_iter`` parameter settings (lists or ``rvs`` callables)."""

    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring=None,
                 refit=True, random_state=0):
        super().__init__(estimator, param_distributions, cv, scoring, refit, random_state)
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def _candidates(self):
        rng = np.random.RandomState(self.random_state)
        out = []
        for _ in range(self.n_iter):
            p = {}
            for k, v in self.param_distributions.items():
                p[k] = v(rng) if callable(v) else v[rng.randint(len(v))]
            out.append(p)
        return out


def learning_curve(estimator, X, y, train_sizes=None, cv=5, scoring=None, random_state=0):
    """Train/validation score as a function of the training-set size."""
    X, y = np.asarray(X), np.asarray(y)
    train_sizes = np.linspace(0.1, 1.0, 5) if train_sizes is None else np.asarray(train_sizes)
    scorer = get_scorer(scoring)
    splits = list(check_cv(cv, y, estimator, random_state).split(X, y))
    max_train = min(len(tr) for tr, _ in splits)
    sizes = (train_sizes * max_train).astype(int) if train_sizes.dtype.kind == "f" else train_sizes
    sizes = np.maximum(sizes, 2)
    tr_scores, va_scores = np.zeros((len(sizes), len(splits))), np.zeros((len(sizes), len(splits)))
    for j, (tr, te) in enumerate(splits):
        for i, s in enumerate(sizes):
            sub = tr[:s]
            m = estimator.clone().fit(X[sub], y[sub])
            tr_scores[i, j] = scorer(m, X[sub], y[sub])
            va_scores[i, j] = scorer(m, X[te], y[te])
    return sizes, tr_scores, va_scores


def validation_curve(estimator, X, y, param_name, param_range, cv=5, scoring=None, random_state=0):
    X, y = np.asarray(X), np.asarray(y)
    scorer = get_scorer(scoring)
    splits = list(check_cv(cv, y, estimator, random_state).split(X, y))
    tr_scores = np.zeros((len(param_range), len(splits)))
    va_scores = np.zeros_like(tr_scores)
    for i, v in enumerate(param_range):
        for j, (tr, te) in enumerate(splits):
            m = estimator.clone().set_params(**{param_name: v}).fit(X[tr], y[tr])
            tr_scores[i, j] = scorer(m, X[tr], y[tr])
            va_scores[i, j] = scorer(m, X[te], y[te])
    return tr_scores, va_scores
