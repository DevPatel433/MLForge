"""Regression metrics."""
import numpy as np


def _prep(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def r2_score(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    eps = np.finfo(float).eps
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), eps)))


def median_absolute_error(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    return float(np.median(np.abs(y_true - y_pred)))


def max_error(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def explained_variance_score(y_true, y_pred):
    y_true, y_pred = _prep(y_true, y_pred)
    var_y = np.var(y_true)
    return float(1 - np.var(y_true - y_pred) / var_y) if var_y > 0 else 0.0
