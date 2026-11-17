"""First-order optimizers.

Every optimizer exposes ``step(grad) -> update`` where *update* is the vector
to **subtract** from the parameters. Optimizers are stateful (momentum, moving
averages ...) so models deep-copy them at the start of each ``fit``.

All optimizers accept an optional learning-rate ``schedule`` (see
:mod:`mlforge.optimization.schedules`): a callable ``schedule(t) -> factor``
that multiplies the base learning rate at step ``t``.
"""
import numpy as np


class Optimizer:
    def __init__(self, learning_rate=0.01, schedule=None):
        self.lr = learning_rate
        self.schedule = schedule
        self.t = 0

    @property
    def learning_rate(self):
        return self.lr

    def _lr(self):
        """Current learning rate. ``self.t`` has already been incremented by
        ``step``, so the schedule sees the zero-based index of this step."""
        factor = self.schedule(max(self.t - 1, 0)) if self.schedule is not None else 1.0
        return self.lr * factor

    def step(self, grad):  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self):
        self.t = 0
        for name in ("velocity", "cache", "m", "v"):
            if hasattr(self, name):
                setattr(self, name, None)

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items()
                         if k in ("lr", "beta", "beta1", "beta2", "epsilon", "schedule"))
        return f"{type(self).__name__}({args})"


class GradientDescent(Optimizer):
    """Vanilla gradient descent: ``theta <- theta - lr * grad``.

    Used for batch, mini-batch and stochastic GD alike: the difference lies in
    how many samples the *caller* uses to compute ``grad``.
    """

    def step(self, grad):
        self.t += 1
        return self._lr() * grad


class SGD(GradientDescent):
    """Alias of :class:`GradientDescent` kept for readability."""


class Momentum(Optimizer):
    """Classical (heavy-ball) momentum: ``v <- beta v + lr g``."""

    def __init__(self, learning_rate=0.01, beta=0.9, schedule=None):
        super().__init__(learning_rate, schedule)
        self.beta = beta
        self.velocity = None

    def step(self, grad):
        self.t += 1
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)
        self.velocity = self.beta * self.velocity + self._lr() * grad
        return self.velocity


class NesterovMomentum(Optimizer):
    """Nesterov accelerated gradient in the "look-ahead free" formulation
    (Sutskever et al., 2013), which needs the gradient at the current point only:

        v_new = beta v + lr g
        update = beta v_new + lr g
    """

    def __init__(self, learning_rate=0.01, beta=0.9, schedule=None):
        super().__init__(learning_rate, schedule)
        self.beta = beta
        self.velocity = None

    def step(self, grad):
        self.t += 1
        lr = self._lr()
        if self.velocity is None:
            self.velocity = np.zeros_like(grad)
        self.velocity = self.beta * self.velocity + lr * grad
        return self.beta * self.velocity + lr * grad


class AdaGrad(Optimizer):
    """AdaGrad: per-parameter rates shrink with accumulated squared gradients."""

    def __init__(self, learning_rate=0.1, epsilon=1e-8, schedule=None):
        super().__init__(learning_rate, schedule)
        self.epsilon = epsilon
        self.cache = None

    def step(self, grad):
        self.t += 1
        if self.cache is None:
            self.cache = np.zeros_like(grad)
        self.cache += grad ** 2
        return self._lr() * grad / (np.sqrt(self.cache) + self.epsilon)


class RMSProp(Optimizer):
    """RMSProp: divide by a moving average of squared gradients."""

    def __init__(self, learning_rate=0.01, beta=0.9, epsilon=1e-8, schedule=None):
        super().__init__(learning_rate, schedule)
        self.beta = beta
        self.epsilon = epsilon
        self.cache = None

    def step(self, grad):
        self.t += 1
        if self.cache is None:
            self.cache = np.zeros_like(grad)
        self.cache = self.beta * self.cache + (1 - self.beta) * grad ** 2
        return self._lr() * grad / (np.sqrt(self.cache) + self.epsilon)


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2014) with bias-corrected moment estimates."""

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 schedule=None):
        super().__init__(learning_rate, schedule)
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m = None
        self.v = None

    def step(self, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self._lr() * m_hat / (np.sqrt(v_hat) + self.epsilon)


OPTIMIZERS = {
    "gd": GradientDescent, "sgd": SGD, "momentum": Momentum,
    "nesterov": NesterovMomentum, "adagrad": AdaGrad, "rmsprop": RMSProp, "adam": Adam,
}


def get_optimizer(optimizer, learning_rate=0.01):
    """Resolve a string / class / instance into a fresh optimizer instance."""
    import copy
    if optimizer is None:
        return GradientDescent(learning_rate)
    if isinstance(optimizer, str):
        return OPTIMIZERS[optimizer.lower()](learning_rate)
    if isinstance(optimizer, type):
        return optimizer(learning_rate)
    return copy.deepcopy(optimizer)
