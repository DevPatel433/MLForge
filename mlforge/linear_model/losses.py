"""Loss functions with analytic gradients for the linear models.

All functions take a *flat* parameter vector ``theta = [w_1 .. w_d, b]`` so
that they plug straight into :func:`mlforge.optimization.minimize_batches`
and into :func:`mlforge.utils.check_gradient`.
"""
import numpy as np

from ..utils.math_utils import sigmoid, softmax


def mse_loss_grad(theta, X, y, alpha=0.0, fit_intercept=True):
    """Mean squared error (+ optional L2 ``alpha/n * ||w||^2``) and gradient.

    L = (1/n) sum (Xw + b - y)^2 + (alpha/n) ||w||^2
    dL/dw = (2/n) X^T e + (2 alpha / n) w,   dL/db = (2/n) sum e
    """
    n = X.shape[0]
    w, b = theta[:-1], theta[-1]
    err = X @ w + b - y
    loss = np.mean(err ** 2) + alpha / n * np.dot(w, w)
    gw = 2.0 / n * (X.T @ err) + 2.0 * alpha / n * w
    gb = 2.0 / n * err.sum() if fit_intercept else 0.0
    return loss, np.append(gw, gb)


def logistic_loss_grad(theta, X, y, C=None, fit_intercept=True):
    """Binary cross-entropy (+ optional L2) and gradient, ``y`` in {0, 1}.

    L = -mean[y log p + (1-y) log(1-p)] + (1 / (2 C n)) ||w||^2,  p = sigmoid(Xw + b)
    dL/dw = (1/n) X^T (p - y) + w / (C n),        dL/db = mean(p - y)
    """
    n = X.shape[0]
    w, b = theta[:-1], theta[-1]
    z = X @ w + b
    p = sigmoid(z)
    # numerically stable log-loss: log(1+e^z) - y z
    loss = np.mean(np.logaddexp(0, z) - y * z)
    gw = X.T @ (p - y) / n
    if C is not None:
        loss += np.dot(w, w) / (2.0 * C * n)
        gw = gw + w / (C * n)
    gb = np.mean(p - y) if fit_intercept else 0.0
    return loss, np.append(gw, gb)


def softmax_loss_grad(theta, X, Y, C=None, fit_intercept=True):
    """Multinomial cross-entropy (+ optional L2) and gradient.

    ``theta`` packs ``W`` (d x k, row-major) followed by ``b`` (k).
    ``Y`` is the one-hot target matrix (n x k).
    """
    n, d = X.shape
    k = Y.shape[1]
    W = theta[: d * k].reshape(d, k)
    b = theta[d * k:]
    logits = X @ W + b
    P = softmax(logits, axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = -np.mean(np.sum(Y * log_p, axis=1))
    G = (P - Y) / n
    gW = X.T @ G
    if C is not None:
        loss += np.sum(W ** 2) / (2.0 * C * n)
        gW = gW + W / (C * n)
    gb = G.sum(axis=0) if fit_intercept else np.zeros(k)
    return loss, np.concatenate([gW.ravel(), gb])
