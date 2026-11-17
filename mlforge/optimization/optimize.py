"""Generic optimisation drivers shared by every gradient-based model.

``minimize_batches`` is the training loop behind linear/logistic/softmax
regression and the linear SVM: it handles full-batch, mini-batch and
stochastic gradient descent, early stopping on loss plateau, optimizer
state reset, and records the loss history.
"""
import numpy as np

from .optimizers import get_optimizer


def minimize_batches(loss_grad, theta0, optimizer=None, n_iters=1000, n_samples=None,
                     batch_size=None, tol=0.0, random_state=None, shuffle=True,
                     learning_rate=0.01, patience=1):
    """Minimise a loss with (mini-batch) gradient descent.

    Parameters
    ----------
    loss_grad : callable ``(theta, idx) -> (loss, grad)``
        ``idx`` is ``None`` for the full dataset or an integer index array
        selecting a mini-batch.
    theta0 : ndarray - initial (flattened) parameters.
    optimizer : str | Optimizer | None - see :func:`get_optimizer`.
    n_iters : int - number of epochs (full passes over the data).
    batch_size : int | None - ``None`` means full batch, ``1`` stochastic GD.
    tol : float - stop when the full-data loss improves by less than ``tol``
        for ``patience`` consecutive epochs (``0`` disables early stopping).

    Returns
    -------
    theta, history (list of per-epoch full-data losses)
    """
    opt = get_optimizer(optimizer, learning_rate)
    if hasattr(opt, "reset"):
        opt.reset()
    theta = np.array(theta0, dtype=float)
    rng = np.random.RandomState(random_state)
    history, stalled = [], 0
    minibatch = batch_size is not None and n_samples is not None and batch_size < n_samples

    for _ in range(n_iters):
        if minibatch:
            order = rng.permutation(n_samples) if shuffle else np.arange(n_samples)
            for start in range(0, n_samples, batch_size):
                idx = order[start:start + batch_size]
                _, grad = loss_grad(theta, idx)
                theta = theta - opt.step(grad)
            loss, _ = loss_grad(theta, None)
        else:
            loss, grad = loss_grad(theta, None)
            theta = theta - opt.step(grad)
            if not np.isfinite(loss):
                break
        if not np.isfinite(loss):
            break
        history.append(float(loss))
        if tol > 0 and len(history) > 1:
            stalled = stalled + 1 if abs(history[-2] - history[-1]) < tol else 0
            if stalled >= patience:
                break
    return theta, history


def minimize(fun, grad, x0, optimizer="gd", n_iters=200, learning_rate=0.01):
    """Minimise an arbitrary smooth function and return the whole trajectory.

    Handy for teaching and for the optimizer-race plots: returns an array of
    shape ``(n_steps + 1, dim)`` with the visited points.
    """
    opt = get_optimizer(optimizer, learning_rate)
    x = np.array(x0, dtype=float)
    path = [x.copy()]
    for _ in range(n_iters):
        g = np.asarray(grad(x), dtype=float)
        x = x - opt.step(g)
        if not np.all(np.isfinite(x)):
            break
        path.append(x.copy())
    return np.array(path)


# ---------------------------------------------------------------- test surfaces
def rosenbrock(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def rosenbrock_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])


def beale(p):
    x, y = p
    return ((1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2)


def beale_grad(p):
    x, y = p
    t1, t2, t3 = 1.5 - x + x * y, 2.25 - x + x * y ** 2, 2.625 - x + x * y ** 3
    return np.array([
        2 * t1 * (y - 1) + 2 * t2 * (y ** 2 - 1) + 2 * t3 * (y ** 3 - 1),
        2 * t1 * x + 2 * t2 * 2 * x * y + 2 * t3 * 3 * x * y ** 2,
    ])


def quadratic_bowl(p, a=1.0, b=10.0):
    return 0.5 * (a * p[0] ** 2 + b * p[1] ** 2)


def quadratic_bowl_grad(p, a=1.0, b=10.0):
    return np.array([a * p[0], b * p[1]])
