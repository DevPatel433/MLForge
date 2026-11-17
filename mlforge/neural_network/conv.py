"""Convolutional layers for the from-scratch neural network (NCHW layout).

The convolution is implemented with **im2col**: every receptive field is unrolled into a row of a
matrix, so the whole layer becomes ONE matrix multiplication (BLAS) instead of nested loops.

    forward:   cols = im2col(x)                 (N*Ho*Wo, C*kh*kw)
               out  = cols @ W.reshape(F, -1).T + b
    backward:  dW   = grad2.T @ cols            dcols = grad2 @ W2        dx = col2im(dcols)
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .layers import Layer


def _pair(v):
    return (v, v) if isinstance(v, int) else tuple(v)


class Conv2D(Layer):
    """2-D cross-correlation ``y[n,f,i,j] = b[f] + sum_{c,u,v} W[f,c,u,v] * x[n,c,i*s+u,j*s+v]``.

    Parameters: ``in_channels``, ``out_channels``, ``kernel_size``, ``stride``, ``padding``
    (zero padding), ``init`` ('he' | 'xavier' | 'normal').
    """
    trainable = True

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, init="he", random_state=None):
        self.kh, self.kw = _pair(kernel_size)
        self.stride, self.padding = stride, padding
        fan_in = in_channels * self.kh * self.kw
        fan_out = out_channels * self.kh * self.kw
        scale = {"he": np.sqrt(2.0 / fan_in), "xavier": np.sqrt(2.0 / (fan_in + fan_out)), "normal": 0.01}.get(init)
        if scale is None:
            raise ValueError("init must be 'he', 'xavier' or 'normal'")
        rng = np.random.RandomState(random_state)
        self.W = rng.randn(out_channels, in_channels, self.kh, self.kw) * scale
        self.b = np.zeros(out_channels)
        self.dW, self.db = np.zeros_like(self.W), np.zeros_like(self.b)

    def forward(self, x, training=True):
        N, C, H, W_ = x.shape
        p, s = self.padding, self.stride
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        win = sliding_window_view(xp, (self.kh, self.kw), axis=(2, 3))[:, :, ::s, ::s]     # (N,C,Ho,Wo,kh,kw)
        Ho, Wo = win.shape[2], win.shape[3]
        self._cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, -1)
        self._shape = (x.shape, xp.shape, Ho, Wo)
        F = self.W.shape[0]
        out = self._cols @ self.W.reshape(F, -1).T + self.b
        return out.reshape(N, Ho, Wo, F).transpose(0, 3, 1, 2)

    def backward(self, grad):
        (N, C, H, W_), xp_shape, Ho, Wo = self._shape
        F = self.W.shape[0]
        g2 = grad.transpose(0, 2, 3, 1).reshape(-1, F)                                   # (N*Ho*Wo, F)
        self.dW = (g2.T @ self._cols).reshape(self.W.shape)
        self.db = g2.sum(axis=0)
        dcols = (g2 @ self.W.reshape(F, -1)).reshape(N, Ho, Wo, C, self.kh, self.kw)
        dxp = np.zeros(xp_shape)
        s = self.stride
        for u in range(self.kh):                                                         # col2im: scatter-add each tap
            for v in range(self.kw):
                dxp[:, :, u:u + s * Ho:s, v:v + s * Wo:s] += dcols[:, :, :, :, u, v].transpose(0, 3, 1, 2)
        p = self.padding
        return dxp[:, :, p:p + H, p:p + W_] if p else dxp

    def params(self):
        return [self.W, self.b]

    def grads(self):
        return [self.dW, self.db]


class MaxPool2D(Layer):
    """Non-overlapping max pooling (stride == pool size). Trailing rows / columns that do not fill a
    window are dropped. Ties are broken towards the first maximum so that exactly one input per
    window receives the gradient."""

    def __init__(self, pool_size=2):
        self.k = pool_size

    def forward(self, x, training=True):
        N, C, H, W = x.shape
        k = self.k
        Ho, Wo = H // k, W // k
        win = x[:, :, :Ho * k, :Wo * k].reshape(N, C, Ho, k, Wo, k).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, Ho, Wo, k * k)
        self._arg = win.argmax(axis=-1)
        self._shape = (x.shape, Ho, Wo)
        return np.take_along_axis(win, self._arg[..., None], axis=-1)[..., 0]

    def backward(self, grad):
        (N, C, H, W), Ho, Wo = self._shape
        k = self.k
        win = np.zeros((N, C, Ho, Wo, k * k))
        np.put_along_axis(win, self._arg[..., None], grad[..., None], axis=-1)
        dx = np.zeros((N, C, H, W))
        dx[:, :, :Ho * k, :Wo * k] = win.reshape(N, C, Ho, Wo, k, k).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, Ho * k, Wo * k)
        return dx


class Flatten(Layer):
    """(N, ...) -> (N, prod(...)), for connecting convolutional stacks to Dense layers."""

    def forward(self, x, training=True):
        self._shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, grad):
        return grad.reshape(self._shape)
