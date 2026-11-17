"""NumPy-only synthetic dataset generators for demos, tests and benchmarks."""
import numpy as np


def _rng(random_state):
    return np.random.RandomState(random_state)


def make_blobs(n_samples=200, centers=3, n_features=2, cluster_std=1.0,
               center_box=(-8.0, 8.0), random_state=None):
    rng = _rng(random_state)
    if isinstance(centers, int):
        centers = rng.uniform(center_box[0], center_box[1], size=(centers, n_features))
    centers = np.asarray(centers, dtype=float)
    k = len(centers)
    y = np.arange(n_samples) % k
    X = centers[y] + rng.randn(n_samples, centers.shape[1]) * cluster_std
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_moons(n_samples=200, noise=0.1, random_state=None):
    rng = _rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    t0 = np.linspace(0, np.pi, n0)
    t1 = np.linspace(0, np.pi, n1)
    outer = np.c_[np.cos(t0), np.sin(t0)]
    inner = np.c_[1 - np.cos(t1), 0.5 - np.sin(t1)]
    X = np.vstack([outer, inner]) + rng.randn(n_samples, 2) * noise
    y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=None):
    rng = _rng(random_state)
    n0 = n_samples // 2
    n1 = n_samples - n0
    t0 = np.linspace(0, 2 * np.pi, n0, endpoint=False)
    t1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    X = np.vstack([np.c_[np.cos(t0), np.sin(t0)], factor * np.c_[np.cos(t1), np.sin(t1)]])
    X += rng.randn(n_samples, 2) * noise
    y = np.r_[np.zeros(n0, dtype=int), np.ones(n1, dtype=int)]
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_spirals(n_samples=300, n_classes=3, noise=0.2, random_state=None):
    rng = _rng(random_state)
    per = n_samples // n_classes
    X, y = [], []
    for c in range(n_classes):
        r = np.linspace(0.2, 1.0, per)
        t = np.linspace(c * 4, (c + 1) * 4, per) + rng.randn(per) * noise
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y.append(np.full(per, c))
    X, y = np.vstack(X), np.concatenate(y)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def make_regression(n_samples=100, n_features=1, noise=0.0, coef=False, bias=0.0,
                    random_state=None):
    rng = _rng(random_state)
    X = rng.randn(n_samples, n_features)
    w = rng.uniform(-3, 3, size=n_features)
    y = X @ w + bias + rng.randn(n_samples) * noise
    return (X, y, w) if coef else (X, y)


def make_classification(n_samples=200, n_features=5, n_informative=3, n_classes=2,
                        class_sep=1.5, random_state=None):
    rng = _rng(random_state)
    y = np.arange(n_samples) % n_classes
    X = rng.randn(n_samples, n_features)
    centers = rng.randn(n_classes, n_informative) * class_sep
    X[:, :n_informative] += centers[y]
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]
