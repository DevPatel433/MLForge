"""Numerical gradient checking.

The single most useful debugging tool when implementing learning algorithms
from scratch: compare an analytic gradient with a central finite-difference
estimate. MLForge's own test-suite uses it to *prove* that the gradients of
its models are correct.
"""
import numpy as np


def numerical_gradient(f, x, eps=1e-6):
    """Central-difference gradient of scalar function ``f`` at ``x``."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        plus = f(x)
        x[idx] = orig - eps
        minus = f(x)
        x[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = np.maximum(np.linalg.norm(a) + np.linalg.norm(b), 1e-12)
    return np.linalg.norm(a - b) / denom


def check_gradient(f, grad_f, x, eps=1e-6, tol=1e-5):
    """Return ``(passed, relative_error)`` comparing ``grad_f(x)`` to the
    numerical gradient of ``f``."""
    x = np.array(x, dtype=float)
    analytic = np.asarray(grad_f(x.copy()), dtype=float)
    numeric = numerical_gradient(f, x.copy(), eps)
    err = relative_error(analytic, numeric)
    return err < tol, err
