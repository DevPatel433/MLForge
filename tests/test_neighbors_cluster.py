import numpy as np
import pytest

from mlforge.cluster import DBSCAN, AgglomerativeClustering, GaussianMixture, KMeans, elbow_curve, fcluster, linkage, silhouette_curve
from mlforge.datasets import make_blobs, make_moons
from mlforge.metrics import adjusted_rand_score
from mlforge.neighbors import KNeighborsClassifier, KNeighborsRegressor


def test_knn_exact_on_toy_problem():
    X = np.array([[0, 0], [0, 1], [5, 5], [5, 6]], float)
    y = np.array([0, 0, 1, 1])
    m = KNeighborsClassifier(k=1).fit(X, y)
    assert m.predict([[0, 0.4], [5, 5.6]]).tolist() == [0, 1]
    dist, idx = m.kneighbors([[0, 0.1]], k=2)
    assert idx[0].tolist() == [0, 1] and dist[0, 0] == pytest.approx(0.1)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev", "cosine", "minkowski"])
def test_knn_metrics(blobs3, metric):
    X, y = blobs3
    assert KNeighborsClassifier(5, metric=metric).fit(X, y).score(X, y) > 0.9


def test_knn_distance_weighting_and_exact_match():
    X = np.array([[0.0], [1.0], [10.0]])
    y = np.array([0, 1, 1])
    uni = KNeighborsClassifier(3).fit(X, y).predict([[0.0]])
    dist = KNeighborsClassifier(3, weights="distance").fit(X, y).predict([[0.0]])
    assert uni[0] == 1 and dist[0] == 0
    assert KNeighborsClassifier(3, weighted=True).fit(X, y).weights == "distance"


def test_knn_regressor_and_invalid_k():
    X = np.arange(4.0).reshape(-1, 1)
    assert KNeighborsRegressor(2).fit(X, X.ravel()).predict([[1.5]])[0] == pytest.approx(1.5)
    assert KNeighborsRegressor(2, weights="distance").fit(X, X.ravel()).predict([[1.0]])[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        KNeighborsClassifier(10).fit(X, [0, 1, 0, 1])
    with pytest.raises(ValueError):
        KNeighborsClassifier(0).fit(X, [0, 1, 0, 1])


def test_kmeans_recovers_blobs():
    X, y = make_blobs(300, 3, cluster_std=0.6, random_state=2)
    km = KMeans(3, random_state=0).fit(X)
    assert adjusted_rand_score(y, km.labels_) > 0.95
    assert km.inertia_ == pytest.approx(sum(((X[km.labels_ == k] - km.cluster_centers_[k]) ** 2).sum() for k in range(3)))
    assert np.array_equal(km.predict(X), km.labels_) and km.transform(X).shape == (300, 3)
    assert km.fit_predict(X).shape == (300,) and km.centroids.shape == (3, 2)


@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_kmeans_history_is_monotone_in_inertia(init):
    X, _ = make_blobs(200, 4, random_state=1)
    km = KMeans(4, init=init, n_init=1, random_state=0).fit(X)
    inertias = [((X[:, None] - c[None]) ** 2).sum(-1).min(1).sum() for c in km.history]
    assert np.all(np.diff(inertias) <= 1e-8) and len(km.history) == km.n_iter_ + 1


def test_kmeans_more_restarts_never_worse():
    X, _ = make_blobs(300, 5, cluster_std=1.5, random_state=3)
    one = KMeans(5, init="random", n_init=1, random_state=0).fit(X).inertia_
    many = KMeans(5, init="random", n_init=15, random_state=0).fit(X).inertia_
    assert many <= one + 1e-9


def test_kmeans_edge_cases():
    with pytest.raises(ValueError):
        KMeans(5).fit(np.ones((3, 2)))
    km = KMeans(2, random_state=0).fit(np.ones((6, 2)))      # identical points / empty clusters
    assert np.isfinite(km.inertia_)


def test_elbow_and_silhouette_curves_pick_true_k():
    X, _ = make_blobs(240, [[-8, -8], [8, -8], [-8, 8], [8, 8]], cluster_std=0.6, random_state=5)
    ks = range(2, 8)
    assert np.all(np.diff(elbow_curve(X, ks)) < 0)
    assert list(ks)[int(np.argmax(silhouette_curve(X, ks)))] == 4


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_hierarchical_clustering(method):
    X, y = make_blobs(60, [[-6, 0], [6, 0], [0, 8]], cluster_std=0.5, random_state=0)
    ac = AgglomerativeClustering(3, method).fit(X)
    assert adjusted_rand_score(y, ac.labels_) == pytest.approx(1.0)
    Z = ac.linkage_matrix_
    assert Z.shape == (59, 4) and np.all(np.diff(Z[:, 2]) >= -1e-9) and Z[-1, 3] == 60
    assert len(np.unique(fcluster(Z, 2))) == 2


def test_linkage_invalid_method():
    with pytest.raises(ValueError):
        linkage(np.random.rand(5, 2), "nope")


def test_dbscan_finds_moons_and_noise():
    X, y = make_moons(300, 0.05, random_state=0)
    db = DBSCAN(0.2, 5).fit(X)
    assert db.n_clusters_ == 2 and adjusted_rand_score(y, db.labels_) > 0.95
    X2 = np.vstack([X, [[5, 5]]])
    assert DBSCAN(0.2, 5).fit_predict(X2)[-1] == -1


def test_gaussian_mixture_em():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(200, 2), rng.randn(200, 2) * 0.5 + [5, 5]])
    g = GaussianMixture(2, random_state=0).fit(X)
    assert np.all(np.diff(g.log_likelihood_history_) > -1e-8)           # EM never decreases the likelihood
    assert sorted(np.round(g.weights_, 1)) == [0.5, 0.5]
    assert np.allclose(g.predict_proba(X).sum(1), 1) and g.n_iter_ >= 1
    one = GaussianMixture(1, random_state=0).fit(X)
    assert g.bic(X) < one.bic(X) and g.aic(X) < one.aic(X)
    S, comp = g.sample(50, random_state=0)
    assert S.shape == (50, 2) and comp.shape == (50,)
    assert g.score(X) > one.score(X)


def test_knn_tie_breaking_is_deterministic():
    X = np.array([[0.0], [1.0], [-1.0], [1.0], [-1.0]])        # three-way tie at distance 1
    y = np.array([0, 1, 2, 1, 2])
    d, idx = KNeighborsClassifier(2).fit(X, y).kneighbors([[0.0]])
    assert idx[0].tolist() == [0, 1] and d[0].tolist() == [0.0, 1.0]       # lowest index wins the tie
    runs = {tuple(KNeighborsClassifier(3).fit(X, y).kneighbors([[0.0]])[1][0]) for _ in range(5)}
    assert len(runs) == 1
