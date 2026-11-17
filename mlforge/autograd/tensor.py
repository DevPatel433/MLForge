"""A compact reverse-mode automatic differentiation engine on top of NumPy.

    >>> from mlforge.autograd import Tensor
    >>> x = Tensor([2.0, 3.0], requires_grad=True)
    >>> y = ((x * x).sum() + x.exp().sum())
    >>> y.backward()
    >>> x.grad            # 2x + e^x
    array([11.3890561, 26.0855369])

Supports broadcasting, matrix multiplication, reductions and the activation
functions needed to build a neural network from nothing but this file.
"""
import numpy as np

_GRAD_ENABLED = True


class no_grad:
    """Context manager that disables graph construction (inference mode)."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev, _GRAD_ENABLED = _GRAD_ENABLED, False

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev


def _unbroadcast(grad, shape):
    """Sum ``grad`` over the axes that were broadcast to reach its shape."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None, _op=""):
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        self._parents = _parents
        self._backward = _backward
        self._op = _op

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _wrap(x):
        return x if isinstance(x, Tensor) else Tensor(x)

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    def _make(self, data, parents, backward, op):
        needs = _GRAD_ENABLED and any(p.requires_grad for p in parents)
        out = Tensor(data, requires_grad=needs,
                     _parents=parents if needs else (),
                     _backward=backward if needs else None, _op=op)
        return out

    def _accum(self, g):
        if not self.requires_grad:
            return
        g = _unbroadcast(g, self.data.shape)
        self.grad = g if self.grad is None else self.grad + g

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = self._wrap(other)
        out = self._make(self.data + other.data, (self, other), None, "+")

        def _b():
            self._accum(out.grad)
            other._accum(out.grad)
        out._backward = _b if out.requires_grad else None
        return out

    def __mul__(self, other):
        other = self._wrap(other)
        out = self._make(self.data * other.data, (self, other), None, "*")

        def _b():
            self._accum(out.grad * other.data)
            other._accum(out.grad * self.data)
        out._backward = _b if out.requires_grad else None
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._wrap(other))

    def __rsub__(self, other):
        return self._wrap(other) + (-self)

    def __truediv__(self, other):
        return self * self._wrap(other) ** -1.0

    def __rtruediv__(self, other):
        return self._wrap(other) * self ** -1.0

    __radd__ = __add__
    __rmul__ = __mul__

    def __pow__(self, p):
        assert isinstance(p, (int, float)), "only scalar exponents are supported"
        out = self._make(self.data ** p, (self,), None, f"**{p}")

        def _b():
            self._accum(out.grad * p * self.data ** (p - 1))
        out._backward = _b if out.requires_grad else None
        return out

    def __matmul__(self, other):
        other = self._wrap(other)
        out = self._make(self.data @ other.data, (self, other), None, "@")

        def _b():
            g = out.grad
            a, b = self.data, other.data
            if a.ndim == 1 and b.ndim == 1:
                self._accum(g * b)
                other._accum(g * a)
                return
            a2 = a if a.ndim > 1 else a[None, :]
            b2 = b if b.ndim > 1 else b[:, None]
            g2 = g
            if a.ndim == 1:
                g2 = g2[None, :]
            if b.ndim == 1:
                g2 = g2[:, None]
            ga = g2 @ b2.swapaxes(-1, -2)
            gb = a2.swapaxes(-1, -2) @ g2
            self._accum(ga.reshape(a.shape) if a.ndim == 1 else ga)
            other._accum(gb.reshape(b.shape) if b.ndim == 1 else gb)
        out._backward = _b if out.requires_grad else None
        return out

    # -------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        out = self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), None, "sum")

        def _b():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accum(np.broadcast_to(g, self.data.shape).copy())
        out._backward = _b if out.requires_grad else None
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod(
            [self.data.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # ------------------------------------------------------------- elementwise
    def exp(self):
        out = self._make(np.exp(self.data), (self,), None, "exp")
        out._backward = (lambda: self._accum(out.grad * out.data)) if out.requires_grad else None
        return out

    def log(self):
        out = self._make(np.log(self.data), (self,), None, "log")
        out._backward = (lambda: self._accum(out.grad / self.data)) if out.requires_grad else None
        return out

    def relu(self):
        out = self._make(np.maximum(self.data, 0), (self,), None, "relu")
        out._backward = (lambda: self._accum(out.grad * (self.data > 0))) if out.requires_grad else None
        return out

    def tanh(self):
        out = self._make(np.tanh(self.data), (self,), None, "tanh")
        out._backward = (lambda: self._accum(out.grad * (1 - out.data ** 2))) if out.requires_grad else None
        return out

    def sigmoid(self):
        s = 1 / (1 + np.exp(-np.clip(self.data, -500, 500)))
        out = self._make(s, (self,), None, "sigmoid")
        out._backward = (lambda: self._accum(out.grad * s * (1 - s))) if out.requires_grad else None
        return out

    def log_softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out = self._make(shifted - lse, (self,), None, "log_softmax")

        def _b():
            p = np.exp(out.data)
            g = out.grad
            self._accum(g - p * g.sum(axis=axis, keepdims=True))
        out._backward = _b if out.requires_grad else None
        return out

    def reshape(self, *shape):
        out = self._make(self.data.reshape(*shape), (self,), None, "reshape")
        out._backward = (lambda: self._accum(out.grad.reshape(self.data.shape))) if out.requires_grad else None
        return out

    @property
    def T(self):
        out = self._make(self.data.T, (self,), None, "T")
        out._backward = (lambda: self._accum(out.grad.T)) if out.requires_grad else None
        return out

    # ---------------------------------------------------------------- backprop
    def zero_grad(self):
        self.grad = None

    def backward(self, grad=None):
        """Reverse-mode differentiation from this tensor through the graph."""
        topo, seen = [], set()

        def build(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._parents:
                build(p)
            topo.append(t)

        # iterative-safe for moderately deep graphs
        import sys
        old = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old, 10000))
        try:
            build(self)
        finally:
            sys.setrecursionlimit(old)

        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=float)
        for t in reversed(topo):
            if t._backward is not None and t.grad is not None:
                t._backward()
