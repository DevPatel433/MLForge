"""Input validation helpers with human-friendly error messages."""
import numpy as np


class NotFittedError(RuntimeError):
    """Raised when an estimator is used before ``fit`` has been called."""


def check_array(X, allow_1d=False, allow_nan=False, dtype=float):
    try:
        X = np.asarray(X, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not convert input to a numeric array: {exc}") from exc
    if X.ndim == 0:
        raise ValueError("Expected a 2D array, got a scalar.")
    if X.ndim == 1:
        if allow_1d:
            return X
        raise ValueError(
            "Expected a 2D array of shape (n_samples, n_features), got a 1D array. "
            "Reshape your data with X.reshape(-1, 1) for a single feature or "
            "X.reshape(1, -1) for a single sample."
        )
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D array, got {X.ndim}D.")
    if X.shape[0] == 0:
        raise ValueError("Found an empty array with 0 samples.")
    if not allow_nan and not np.all(np.isfinite(X)):
        raise ValueError("Input contains NaN or infinity. Impute or drop those values first "
                         "(see mlforge.preprocessing.SimpleImputer).")
    return X


def check_X_y(X, y, allow_1d=False, allow_nan=False, y_numeric=False):
    X = check_array(X, allow_1d=allow_1d, allow_nan=allow_nan)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.ravel()
    if y.ndim != 1:
        raise ValueError(f"`y` must be 1D, got shape {y.shape}.")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y have inconsistent numbers of samples: {X.shape[0]} vs {y.shape[0]}.")
    if y_numeric and not np.issubdtype(y.dtype, np.number):
        raise ValueError("`y` must be numeric.")
    return X, y
