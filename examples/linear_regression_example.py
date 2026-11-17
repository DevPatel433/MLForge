"""Example: fit LinearRegression on a synthetic dataset and plot the result."""
import matplotlib.pyplot as plt
import numpy as np

from mlforge.linear_model import LinearRegression
from mlforge.metrics import r2_score

rng = np.random.RandomState(42)
X = rng.rand(100, 1) * 10
y = 3.5 * X.ravel() + 7 + rng.randn(100) * 2

model = LinearRegression(learning_rate=0.01, n_iters=2000)
model.fit(X, y)
preds = model.predict(X)

print(f"Learned weight: {model.weights[0]:.3f}, bias: {model.bias:.3f}")
print(f"R^2 score: {r2_score(y, preds):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].scatter(X, y, alpha=0.6, label="data")
order = np.argsort(X.ravel())
axes[0].plot(X.ravel()[order], preds[order], color="red", label="fit")
axes[0].set_title("Linear Regression Fit")
axes[0].set_xlabel("X")
axes[0].set_ylabel("y")
axes[0].legend()

axes[1].plot(model.cost_history)
axes[1].set_title("Cost vs. Epoch")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE")

plt.tight_layout()
plt.savefig("linear_regression_example.png", dpi=120)
print("Saved plot to linear_regression_example.png")
