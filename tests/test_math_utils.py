import numpy as np
import pytest

from mlforge.utils import math_utils as mu
from mlforge.utils import pairwise_distances


def test_basic_statistics():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert mu.mean(x) == 2.5
    assert np.isclose(mu.variance(x), 1.25)
    assert np.isclose(mu.std(x), np.sqrt(1.25))
    assert np.isclose(mu.variance(x, ddof=1), 5 / 3)


def test_distances_and_dot():
    a, b = np.array([0.0, 0.0]), np.array([3.0, 4.0])
    assert np.isclose(mu.euclidean_distance(a, b), 5.0)
    assert np.isclose(mu.manhattan_distance(a, b), 7.0)
    assert mu.dot_product(b, b) == 25.0


def test_cosine_similarity_edge_cases():
    assert np.isclose(mu.cosine_similarity([1, 0], [0, 1]), 0.0)
    assert np.isclose(mu.cosine_similarity([1, 2], [2, 4]), 1.0)
    assert mu.cosine_similarity([0, 0], [1, 1]) == 0.0


def test_covariance_correlation_match_numpy(rng):
    X = rng.randn(50, 4)
    assert np.allclose(mu.covariance_matrix(X, ddof=1), np.cov(X.T))
    assert np.allclose(mu.correlation_matrix(X), np.corrcoef(X.T))


def test_sigmoid_softmax_stability():
    assert np.isclose(mu.sigmoid(0), 0.5)
    assert mu.sigmoid(-1000) == pytest.approx(0.0, abs=1e-12)
    p = mu.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isclose(p.sum(), 1.0) and np.all(np.isfinite(p))


@pytest.mark.parametrize("metric", ["euclidean", "sqeuclidean", "manhattan", "chebyshev", "cosine", "minkowski"])
def test_pairwise_distances_vs_scipy_like(rng, metric):
    A, B = rng.randn(7, 3), rng.randn(5, 3)
    D = pairwise_distances(A, B, metric, p=3)
    i, j = 2, 4
    ref = {
        "euclidean": np.linalg.norm(A[i] - B[j]),
        "sqeuclidean": np.sum((A[i] - B[j]) ** 2),
        "manhattan": np.abs(A[i] - B[j]).sum(),
        "chebyshev": np.abs(A[i] - B[j]).max(),
        "cosine": 1 - A[i] @ B[j] / np.linalg.norm(A[i]) / np.linalg.norm(B[j]),
        "minkowski": (np.abs(A[i] - B[j]) ** 3).sum() ** (1 / 3),
    }[metric]
    assert D.shape == (7, 5) and np.isclose(D[i, j], ref)


def test_pairwise_unknown_metric():
    with pytest.raises(ValueError):
        pairwise_distances(np.ones((2, 2)), metric="nope")


# ---- tests added after mutation testing exposed gaps -------------------------------------------
def test_manhattan_distance_with_nonzero_origin_and_signs():
    assert mu.manhattan_distance([1.0, -2.0], [4.0, 2.0]) == pytest.approx(7.0)      # |1-4| + |-2-2|
    assert mu.manhattan_distance([5.0, 5.0], [2.0, 9.0]) == pytest.approx(7.0)


def test_covariance_matrix_default_is_population_covariance(rng):
    X = rng.randn(30, 3)
    assert np.allclose(mu.covariance_matrix(X), np.cov(X.T, ddof=0))
    assert np.allclose(mu.covariance_matrix(X, ddof=1), np.cov(X.T, ddof=1))
    assert not np.allclose(mu.covariance_matrix(X), mu.covariance_matrix(X, ddof=1))


def test_cosine_distance_handles_zero_vectors_without_nan():
    A = np.array([[0.0, 0.0], [1.0, 0.0]])
    B = np.array([[0.0, 0.0], [0.0, 2.0]])
    D = pairwise_distances(A, B, "cosine")
    assert np.all(np.isfinite(D))
    assert D[0, 0] == 1.0 and D[0, 1] == 1.0 and D[1, 0] == 1.0       # a zero vector is "orthogonal" to everything
    assert D[1, 1] == pytest.approx(1.0)                              # (1,0) vs (0,2) are orthogonal


@pytest.mark.parametrize("metric", ["manhattan", "chebyshev", "minkowski"])
def test_chunked_pairwise_distances_equal_the_unchunked_result(rng, monkeypatch, metric):
    import mlforge.utils.pairwise as pw
    A, B = rng.randn(23, 4), rng.randn(11, 4)
    full = pairwise_distances(A, B, metric, p=3)
    monkeypatch.setattr(pw, "_CHUNK_ELEMENTS", 50)                   # force many tiny chunks (step = 1 row)
    assert np.allclose(pairwise_distances(A, B, metric, p=3), full)
    monkeypatch.setattr(pw, "_CHUNK_ELEMENTS", 200)                  # chunks of several rows, last one ragged
    assert np.allclose(pairwise_distances(A, B, metric, p=3), full)
