"""Shared estimator machinery: parameter introspection, cloning, persistence,
input validation hooks and "not fitted" protection.

Every public estimator inherits from :class:`BaseEstimator`. Subclasses get,
for free and without writing any boilerplate:

* ``get_params`` / ``set_params`` / ``clone`` (needed by grid search & CV)
* readable ``repr``
* ``save`` / ``load`` (pickle based)
* automatic validation of ``X`` / ``y`` in ``fit`` and ``predict``-style methods
* a clear :class:`NotFittedError` when predicting before fitting
"""
import copy
import functools
import inspect
import pickle

from .utils.validation import NotFittedError, check_array, check_X_y

_PREDICT_LIKE = ("predict", "predict_proba", "predict_log_proba", "transform",
                 "decision_function", "score", "inverse_transform")


def _wrap_fit(fn):
    if getattr(fn, "_mlforge_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapper(self, X, y=None, *args, **kwargs):
        allow_1d = getattr(self, "_allow_1d", False)
        allow_nan = getattr(self, "_allow_nan", False)
        if y is None:
            X = check_array(X, allow_1d=allow_1d, allow_nan=allow_nan)
            out = fn(self, X, *args, **kwargs)
        else:
            X, y = check_X_y(X, y, allow_1d=allow_1d, allow_nan=allow_nan)
            out = fn(self, X, y, *args, **kwargs)
        self._is_fitted = True
        return out

    wrapper._mlforge_wrapped = True
    return wrapper


def _wrap_predict(fn):
    if getattr(fn, "_mlforge_wrapped", False):
        return fn

    @functools.wraps(fn)
    def wrapper(self, X, *args, **kwargs):
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(
                f"{type(self).__name__} is not fitted yet. Call `fit` before using this method."
            )
        X = check_array(X, allow_1d=getattr(self, "_allow_1d", False),
                        allow_nan=getattr(self, "_allow_nan", False))
        return fn(self, X, *args, **kwargs)

    wrapper._mlforge_wrapped = True
    return wrapper


class BaseEstimator:
    """Base class for all MLForge estimators."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "_validate_io", True):
            return
        if "fit" in cls.__dict__ and callable(cls.__dict__["fit"]):
            cls.fit = _wrap_fit(cls.__dict__["fit"])
        for name in _PREDICT_LIKE:
            if name in cls.__dict__ and callable(cls.__dict__[name]) and name != "score":
                setattr(cls, name, _wrap_predict(cls.__dict__[name]))

    # ------------------------------------------------------------------ params
    @classmethod
    def _param_names(cls):
        sig = inspect.signature(cls.__init__)
        names = []
        for p in sig.parameters.values():
            if p.name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            names.append(p.name)
        # classes that forward **kwargs to a parent (e.g. PolynomialRegression)
        for base in cls.__mro__[1:]:
            if base is object or base is BaseEstimator:
                continue
            if "__init__" in base.__dict__:
                for n in base._param_names():
                    if n not in names:
                        names.append(n)
        return names

    def get_params(self):
        params = {}
        for name in self._param_names():
            if hasattr(self, name):
                params[name] = getattr(self, name)
            elif name == "learning_rate" and hasattr(self, "lr"):
                params[name] = self.lr
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if k == "learning_rate" and not hasattr(self, "learning_rate"):
                self.lr = v
            setattr(self, k, v)
        return self

    def clone(self):
        params = copy.deepcopy(self.get_params())
        return type(self)(**params)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"

    # -------------------------------------------------------------- persistence
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


class ClassifierMixin:
    def score(self, X, y):
        from .metrics.classification import accuracy_score
        return accuracy_score(y, self.predict(X))


class RegressorMixin:
    def score(self, X, y):
        from .metrics.regression import r2_score
        return r2_score(y, self.predict(X))
