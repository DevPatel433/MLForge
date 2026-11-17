"""Train a tiny network on XOR using ONLY the autograd engine (no hand-written backprop)."""
import numpy as np

from mlforge.autograd import Tensor, no_grad

rng = np.random.RandomState(0)
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])
onehot = np.eye(2)[y]

W1, b1 = Tensor(rng.randn(2, 8) * 0.8, requires_grad=True), Tensor(np.zeros(8), requires_grad=True)
W2, b2 = Tensor(rng.randn(8, 2) * 0.8, requires_grad=True), Tensor(np.zeros(2), requires_grad=True)
params = [W1, b1, W2, b2]

for epoch in range(400):
    for p in params:
        p.zero_grad()
    logits = (X @ W1 + b1).tanh() @ W2 + b2
    loss = -(logits.log_softmax(axis=1) * onehot).sum() * (1 / len(y))   # cross-entropy
    loss.backward()                                                      # <- reverse-mode autodiff
    for p in params:
        p.data -= 0.3 * p.grad
    if epoch % 100 == 0:
        print(f"epoch {epoch:3d}  loss {float(loss.data):.4f}")

with no_grad():
    pred = ((X @ W1 + b1).tanh() @ W2 + b2).data.argmax(1)
print("predictions:", pred.tolist(), "  targets:", y.tolist())
