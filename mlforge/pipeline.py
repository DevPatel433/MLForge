"""Chain transformers and a final estimator into a single model."""
import copy

from .base import BaseEstimator


class Pipeline(BaseEstimator):
    """Sequentially apply transformers and a final estimator.

    >>> pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())])
    >>> pipe.fit(X, y).predict(X)

    Step parameters are reachable as ``step__param`` in ``set_params`` and
    therefore in :class:`~mlforge.model_selection.GridSearchCV`.
    """

    _allow_nan = True

    def __init__(self, steps):
        self.steps = steps

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def classes_(self):
        return self.steps[-1][1].classes_

    def get_params(self):
        params = {"steps": self.steps}
        for name, est in self.steps:
            params[name] = est
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if "__" in key:
                name, sub = key.split("__", 1)
                self.named_steps[name].set_params(**{sub: value})
            elif key == "steps":
                self.steps = value
            else:
                self.steps = [(n, value if n == key else e) for n, e in self.steps]
        return self

    def clone(self):
        return Pipeline([(n, e.clone() if hasattr(e, "clone") else copy.deepcopy(e))
                         for n, e in self.steps])

    def __repr__(self):
        inner = ", ".join(f"({n!r}, {e!r})" for n, e in self.steps)
        return f"Pipeline([{inner}])"

    def _transform(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)
        return X

    def fit(self, X, y=None):
        for _, step in self.steps[:-1]:
            X = step.fit_transform(X, y) if _accepts_y(step) else step.fit_transform(X)
        self.steps[-1][1].fit(X, y) if y is not None else self.steps[-1][1].fit(X)
        return self

    def predict(self, X):
        return self.steps[-1][1].predict(self._transform(X))

    def predict_proba(self, X):
        return self.steps[-1][1].predict_proba(self._transform(X))

    def decision_function(self, X):
        return self.steps[-1][1].decision_function(self._transform(X))

    def transform(self, X):
        return self.steps[-1][1].transform(self._transform(X))

    def score(self, X, y):
        return self.steps[-1][1].score(self._transform(X), y)


def _accepts_y(step):
    import inspect
    return "y" in inspect.signature(step.fit_transform).parameters
