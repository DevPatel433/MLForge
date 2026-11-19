import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(0)


@pytest.fixture
def blobs2():
    from mlforge.datasets import make_blobs
    return make_blobs(200, [[-2, -2], [2, 2]], cluster_std=1.0, random_state=0)


@pytest.fixture
def blobs3():
    from mlforge.datasets import make_blobs
    return make_blobs(240, [[-4, 0], [4, 0], [0, 5]], cluster_std=1.0, random_state=1)


@pytest.fixture
def reg_data():
    rng = np.random.RandomState(0)
    X = rng.randn(150, 5)
    w = np.array([3.0, 0.0, -2.0, 0.0, 1.0])
    y = X @ w + 1.5 + rng.randn(150) * 0.3
    return X, y, w
