"""``Sequential`` container: composes layers, computes gradients, trains."""
import copy

import numpy as np

from ..optimization import get_optimizer


class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)

    # --------------------------------------------------------------- forward/back
    def forward(self, x, training=False):
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    __call__ = forward

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    # ------------------------------------------------------------- parameters
    def parameters(self):
        return [p for layer in self.layers for p in layer.params()]

    def gradients(self):
        return [g for layer in self.layers for g in layer.grads()]

    def n_parameters(self):
        return int(sum(p.size for p in self.parameters()))

    def get_flat_params(self):
        return np.concatenate([p.ravel() for p in self.parameters()])

    def set_flat_params(self, vec):
        pos = 0
        for p in self.parameters():
            p[...] = vec[pos:pos + p.size].reshape(p.shape)
            pos += p.size

    def get_flat_grads(self):
        return np.concatenate([g.ravel() for g in self.gradients()])

    def loss_and_grad(self, X, Y, loss, l2=0.0):
        """Full-batch loss and gradients with dropout disabled (for gradient checking)."""
        out = self.forward(X, training=False)
        value = loss.forward(out, Y)
        self.backward(loss.backward())
        if l2:
            for layer in self.layers:
                if layer.trainable:
                    value += 0.5 * l2 * np.sum(layer.W ** 2)
                    layer.dW = layer.dW + l2 * layer.W
        return value, self.get_flat_grads()

    # ----------------------------------------------------------------- training
    def fit(self, X, Y, loss, optimizer="adam", learning_rate=0.01, epochs=100,
            batch_size=32, l2=0.0, validation_data=None, patience=None,
            metric=None, shuffle=True, random_state=None, verbose=0):
        """Mini-batch training. Returns a history dict of per-epoch values
        (``loss``, ``val_loss`` and, if ``metric`` is given, ``metric``/``val_metric``)."""
        rng = np.random.RandomState(random_state)
        n = len(X)
        opts = [get_optimizer(optimizer, learning_rate) for _ in self.parameters()]
        history = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []
        if metric is not None:
            history["metric"] = []
            if validation_data is not None:
                history["val_metric"] = []
        best, best_params, wait = np.inf, None, 0

        for epoch in range(epochs):
            order = rng.permutation(n) if shuffle else np.arange(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                out = self.forward(X[idx], training=True)
                loss.forward(out, Y[idx])
                self.backward(loss.backward())
                for layer in self.layers:
                    if layer.trainable and l2:
                        layer.dW = layer.dW + l2 * layer.W
                for p, g, o in zip(self.parameters(), self.gradients(), opts):
                    p -= o.step(g)

            out = self.forward(X, training=False)
            history["loss"].append(loss.forward(out, Y))
            if metric is not None:
                history["metric"].append(metric(out, Y))
            if validation_data is not None:
                Xv, Yv = validation_data
                outv = self.forward(Xv, training=False)
                vl = loss.forward(outv, Yv)
                history["val_loss"].append(vl)
                if metric is not None:
                    history["val_metric"].append(metric(outv, Yv))
                if patience is not None:
                    if vl < best - 1e-9:
                        best, best_params, wait = vl, copy.deepcopy(self.get_flat_params()), 0
                    else:
                        wait += 1
                        if wait >= patience:
                            if best_params is not None:
                                self.set_flat_params(best_params)
                            break
            if verbose and (epoch + 1) % verbose == 0:
                msg = f"epoch {epoch + 1:4d}  loss {history['loss'][-1]:.4f}"
                if "val_loss" in history:
                    msg += f"  val_loss {history['val_loss'][-1]:.4f}"
                print(msg)
        return history

    def summary(self):
        lines = ["Sequential model", "-" * 48]
        for i, layer in enumerate(self.layers):
            n = sum(p.size for p in layer.params())
            shape = ""
            if hasattr(layer, "kh"):                                           # Conv2D: W is (out, in, kh, kw)
                shape = f"{layer.W.shape[1]} -> {layer.W.shape[0]} ({layer.kh}x{layer.kw})"
            elif hasattr(layer, "Wh"):                                         # LSTM
                shape = f"{layer.Wx.shape[0]} -> {layer.H} (LSTM)"
            elif layer.trainable:                                              # Dense: W is (in, out)
                shape = f"{layer.W.shape[0]} -> {layer.W.shape[1]}"
            lines.append(f"{i:>2} {type(layer).__name__:<12} {shape:<18} {n:>8} params")
        lines += ["-" * 48, f"Total trainable parameters: {self.n_parameters()}"]
        return "\n".join(lines)
