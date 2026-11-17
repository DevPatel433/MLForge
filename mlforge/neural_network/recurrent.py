"""A Long Short-Term Memory layer with full back-propagation through time (BPTT)."""
import numpy as np

from .layers import Layer


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LSTM(Layer):
    """LSTM over inputs of shape ``(N, T, D)``.

    Gates (input ``i``, forget ``f``, candidate ``g``, output ``o``) share one affine map:

        [i f g o] = x_t Wx + h_{t-1} Wh + b
        c_t = sigmoid(f) * c_{t-1} + sigmoid(i) * tanh(g)
        h_t = sigmoid(o) * tanh(c_t)

    ``return_sequences=False`` returns the last hidden state ``(N, H)``; ``True`` returns all
    states ``(N, T, H)``. The forget-gate bias is initialised to 1 (Jozefowicz et al., 2015) so the
    cell remembers by default.
    """
    trainable = True

    def __init__(self, input_size, hidden_size, return_sequences=False, random_state=None):
        rng = np.random.RandomState(random_state)
        H = hidden_size
        self.H, self.return_sequences = H, return_sequences
        self.Wx = rng.randn(input_size, 4 * H) * np.sqrt(1.0 / input_size)
        self.Wh = rng.randn(H, 4 * H) * np.sqrt(1.0 / H)
        self.b = np.zeros(4 * H)
        self.b[H:2 * H] = 1.0
        self.dWx, self.dWh, self.db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)

    def forward(self, x, training=True):
        N, T, _ = x.shape
        H = self.H
        h, c = np.zeros((N, H)), np.zeros((N, H))
        self._cache, hs = [], []
        for t in range(T):
            z = x[:, t] @ self.Wx + h @ self.Wh + self.b
            i, f, g, o = _sigmoid(z[:, :H]), _sigmoid(z[:, H:2 * H]), np.tanh(z[:, 2 * H:3 * H]), _sigmoid(z[:, 3 * H:])
            c_new = f * c + i * g
            tc = np.tanh(c_new)
            h_new = o * tc
            self._cache.append((x[:, t], h, c, i, f, g, o, tc))
            h, c = h_new, c_new
            hs.append(h)
        self._x_shape = x.shape
        return np.stack(hs, axis=1) if self.return_sequences else h

    def backward(self, grad):
        N, T, D = self._x_shape
        H = self.H
        dx = np.zeros(self._x_shape)
        self.dWx, self.dWh, self.db = np.zeros_like(self.Wx), np.zeros_like(self.Wh), np.zeros_like(self.b)
        dh_next, dc_next = np.zeros((N, H)), np.zeros((N, H))
        for t in reversed(range(T)):
            x_t, h_prev, c_prev, i, f, g, o, tc = self._cache[t]
            dh = dh_next + (grad[:, t] if self.return_sequences else (grad if t == T - 1 else 0.0))
            do = dh * tc
            dc = dc_next + dh * o * (1 - tc ** 2)
            di, df, dg = dc * g, dc * c_prev, dc * i
            dz = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
            self.dWx += x_t.T @ dz
            self.dWh += h_prev.T @ dz
            self.db += dz.sum(axis=0)
            dx[:, t] = dz @ self.Wx.T
            dh_next = dz @ self.Wh.T
            dc_next = dc * f
        return dx

    def params(self):
        return [self.Wx, self.Wh, self.b]

    def grads(self):
        return [self.dWx, self.dWh, self.db]
