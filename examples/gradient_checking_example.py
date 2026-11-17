"""Prove that a gradient is correct: analytic gradient vs finite differences.

The single best debugging tool when implementing learning algorithms from scratch.
"""
import numpy as np

from mlforge.linear_model import logistic_loss_grad
from mlforge.neural_network import CrossEntropyLoss, Dense, Sequential, Tanh
from mlforge.utils import check_gradient

rng = np.random.RandomState(0)
X, y = rng.randn(40, 3), rng.randint(0, 2, 40)

# 1) logistic regression loss
theta = rng.randn(4)
ok, err = check_gradient(lambda t: logistic_loss_grad(t, X, y, C=1.0)[0],
                         lambda t: logistic_loss_grad(t, X, y, C=1.0)[1], theta)
print(f"logistic regression gradient correct: {ok}   (relative error {err:.2e})")

# 2) a whole neural network, backprop vs numerical gradient
net = Sequential([Dense(3, 5, "xavier", 0), Tanh(), Dense(5, 2, "xavier", 1)])
loss = CrossEntropyLoss()
f = lambda th: (net.set_flat_params(th), net.loss_and_grad(X, y, loss)[0])[1]
g = lambda th: (net.set_flat_params(th), net.loss_and_grad(X, y, loss)[1])[1]
ok, err = check_gradient(f, g, net.get_flat_params())
print(f"neural-network backprop correct:      {ok}   (relative error {err:.2e})")

# 3) a deliberately WRONG gradient is caught
ok, err = check_gradient(lambda t: float(np.sum(t ** 2)), lambda t: 3 * t, rng.randn(5))
print(f"buggy gradient detected:              {not ok}   (relative error {err:.2e})")
