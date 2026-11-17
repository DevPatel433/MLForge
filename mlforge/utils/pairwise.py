"""Vectorised pairwise distance computations."""
import numpy as np

_CHUNK_ELEMENTS = int(2e7)       # max elements of the (chunk x Y x d) temporary used by the non-BLAS metrics


def pairwise_distances(X, Y=None, metric="euclidean", p=2):
    """Distance matrix between rows of ``X`` and rows of ``Y`` (default ``X``).

    metric: 'euclidean', 'sqeuclidean', 'manhattan', 'chebyshev', 'cosine',
    'minkowski' (with exponent ``p``).
    """
    X = np.asarray(X, dtype=float)
    Y = X if Y is None else np.asarray(Y, dtype=float)
    if metric in ("euclidean", "sqeuclidean"):
        d2 = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2.0 * X @ Y.T
        np.maximum(d2, 0, out=d2)
        if Y is X:
            np.fill_diagonal(d2, 0.0)
        return d2 if metric == "sqeuclidean" else np.sqrt(d2)
    if metric == "cosine":
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        yn = np.linalg.norm(Y, axis=1, keepdims=True)
        xn[xn == 0] = 1
        yn[yn == 0] = 1
        return 1.0 - (X / xn) @ (Y / yn).T
    out = np.empty((X.shape[0], Y.shape[0]))
    step = max(1, int(_CHUNK_ELEMENTS // max(1, Y.shape[0] * X.shape[1])))
    for s in range(0, X.shape[0], step):
        diff = np.abs(X[s:s + step, None, :] - Y[None, :, :])
        if metric == "manhattan":
            out[s:s + step] = diff.sum(-1)
        elif metric == "chebyshev":
            out[s:s + step] = diff.max(-1)
        elif metric == "minkowski":
            out[s:s + step] = (diff ** p).sum(-1) ** (1.0 / p)
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
    return out
