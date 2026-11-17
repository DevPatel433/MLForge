"""Example: K-Means clustering with centroid trajectory visualization."""
import matplotlib.pyplot as plt
import numpy as np

from mlforge.cluster import KMeans

rng = np.random.RandomState(1)
X = np.vstack([
    rng.randn(50, 2) + np.array([0, 0]),
    rng.randn(50, 2) + np.array([6, 6]),
    rng.randn(50, 2) + np.array([-6, 6]),
])

model = KMeans(n_clusters=3, init="k-means++", random_state=1)
model.fit(X)

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap="viridis", alpha=0.6, s=20)

history = np.array(model.history)  # shape (n_iters+1, n_clusters, n_features)
for k in range(model.n_clusters):
    path = history[:, k, :]
    plt.plot(path[:, 0], path[:, 1], "o--", color="red", alpha=0.6, markersize=4)
plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker="X", s=200,
            c="black", label="final centroids")

plt.title("K-Means Clustering (centroid trajectory in red)")
plt.legend()
plt.tight_layout()
plt.savefig("clustering_example.png", dpi=120)
print(f"Final inertia: {model.inertia_:.3f}")
print("Saved plot to clustering_example.png")
