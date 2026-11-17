"""Layers, activations and losses for the from-scratch neural network."""
import numpy as np


class Layer:
    trainable = False

    def forward(self, x, training=True):  # pragma: no cover - interface
        raise NotImplementedError

    def backward(self, grad):  # pragma: no cover - interface
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []


class Dense(Layer):
    """Fully connected layer ``y = x W + b``.

    init : 'he' (ReLU-family), 'xavier' (tanh / sigmoid), 'normal', 'zeros'
    """
    trainable = True

    def __init__(self, n_inputs, n_outputs, init="he", random_state=None):
        rng = np.random.RandomState(random_state)
        if init == "he":
            scale = np.sqrt(2.0 / n_inputs)
        elif init == "xavier":
            scale = np.sqrt(2.0 / (n_inputs + n_outputs))
        elif init == "normal":
            scale = 0.01
        elif init == "zeros":
            scale = 0.0
        else:
            raise ValueError("init must be 'he', 'xavier', 'normal' or 'zeros'")
        self.W = rng.randn(n_inputs, n_outputs) * scale
        self.b = np.zeros((1, n_outputs))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x, training=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self.x.T @ grad
        self.db = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class ReLU(Layer):
    def forward(self, x, training=True):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x, training=True):
        self.mask = x > 0
        return np.where(self.mask, x, self.alpha * x)

    def backward(self, grad):
        return grad * np.where(self.mask, 1.0, self.alpha)


class Sigmoid(Layer):
    def forward(self, x, training=True):
        self.out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh(Layer):
    def forward(self, x, training=True):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out ** 2)


class Softmax(Layer):
    """Softmax layer with the full Jacobian-vector product in ``backward``.
    (For training prefer :class:`CrossEntropyLoss`, which fuses softmax + NLL.)"""

    def forward(self, x, training=True):
        z = x - x.max(axis=1, keepdims=True)
        e = np.exp(z)
        self.out = e / e.sum(axis=1, keepdims=True)
        return self.out

    def backward(self, grad):
        s = self.out
        return s * (grad - np.sum(grad * s, axis=1, keepdims=True))


class Dropout(Layer):
    """Inverted dropout: active only while training, so inference needs no rescaling."""

    def __init__(self, rate=0.5, random_state=None):
        if not 0 <= rate < 1:
            raise ValueError("rate must lie in [0, 1)")
        self.rate = rate
        self.rng = np.random.RandomState(random_state)

    def forward(self, x, training=True):
        if not training or self.rate == 0:
            self.mask = None
            return x
        self.mask = (self.rng.rand(*x.shape) >= self.rate) / (1.0 - self.rate)
        return x * self.mask

    def backward(self, grad):
        return grad if self.mask is None else grad * self.mask


ACTIVATIONS = {"relu": ReLU, "leaky_relu": LeakyReLU, "sigmoid": Sigmoid, "tanh": Tanh}


# ---------------------------------------------------------------------- losses
class CrossEntropyLoss:
    """Softmax + negative log-likelihood on **logits** (numerically stable).
    ``target`` may be integer class ids or a one-hot matrix."""

    def forward(self, logits, target):
        n = logits.shape[0]
        z = logits - logits.max(axis=1, keepdims=True)
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        self.p = np.exp(log_p)
        target = np.asarray(target)
        self.Y = np.eye(logits.shape[1])[target.astype(int)] if target.ndim == 1 else target
        return float(-np.sum(self.Y * log_p) / n)

    def backward(self):
        return (self.p - self.Y) / self.p.shape[0]


class MSELoss:
    def forward(self, pred, target):
        self.diff = pred - np.asarray(target, dtype=float).reshape(pred.shape)
        return float(np.mean(self.diff ** 2))

    def backward(self):
        return 2.0 * self.diff / self.diff.size


class BinaryCrossEntropyLoss:
    """Sigmoid + binary cross-entropy on a single logit per sample."""

    def forward(self, logits, target):
        t = np.asarray(target, dtype=float).reshape(logits.shape)
        self.p = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        self.t = t
        return float(np.mean(np.logaddexp(0, logits) - t * logits))

    def backward(self):
        return (self.p - self.t) / self.t.size
