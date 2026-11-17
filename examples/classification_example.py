"""Example: compare Logistic Regression, Decision Tree, and KNN on a toy
2D classification problem, with decision boundary plots."""
import matplotlib.pyplot as plt
import numpy as np

from mlforge.linear_model import LogisticRegression
from mlforge.metrics import accuracy_score
from mlforge.neighbors import KNeighborsClassifier
from mlforge.tree import DecisionTreeClassifier

rng = np.random.RandomState(0)
X0 = rng.randn(100, 2) - 2
X1 = rng.randn(100, 2) + 2
X = np.vstack([X0, X1])
y = np.array([0] * 100 + [1] * 100)

models = {
    "Logistic Regression": LogisticRegression(learning_rate=0.1, n_iters=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "KNN (k=5)": KNeighborsClassifier(k=5),
}

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),
)
grid = np.c_[xx.ravel(), yy.ravel()]

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X, y)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_title(f"{name}\naccuracy={acc:.3f}")

plt.tight_layout()
plt.savefig("classification_example.png", dpi=120)
print("Saved plot to classification_example.png")
