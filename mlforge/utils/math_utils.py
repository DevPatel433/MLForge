"""Core mathematical utilities used across MLForge (pure NumPy)."""
import numpy as np


def mean(x, axis=None):
    return np.mean(x, axis=axis)


def variance(x, axis=None, ddof=0):
    return np.var(x, axis=axis, ddof=ddof)


def std(x, axis=None, ddof=0):
    return np.std(x, axis=axis, ddof=ddof)


def euclidean_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def manhattan_distance(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.sum(np.abs(a - b), axis=-1)


def dot_product(a, b):
    return np.dot(a, b)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.dot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return num / denom


def covariance_matrix(X, ddof=0):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    n = X.shape[0]
    return (Xc.T @ Xc) / (n - ddof)


def correlation_matrix(X):
    cov = covariance_matrix(X)
    d = np.sqrt(np.diag(cov))
    outer = np.outer(d, d)
    outer[outer == 0] = 1e-12
    return cov / outer


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)
