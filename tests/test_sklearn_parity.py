"""Numerical parity with scikit-learn.

scikit-learn is used *only here and in benchmarks/*, never inside ``mlforge``.
Tolerances are deliberately tight: these tests verify that the from-scratch
implementations compute the same mathematical objects as the reference.
"""
import numpy as np
import pytest

sk = pytest.importorskip("sklearn")
from sklearn import cluster as skc
from sklearn import decomposition as skd
from sklearn import ensemble as ske
from sklearn import linear_model as sklm
from sklearn import metrics as skm
from sklearn import mixture as skmix
from sklearn import naive_bayes as sknb
from sklearn import neighbors as skn
from sklearn import preprocessing as skp
from sklearn import svm as sksvm
from sklearn import tree as skt
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine

import mlforge.linear_model as L
from mlforge import metrics as M
from mlforge.bayes import BernoulliNB, GaussianNB, MultinomialNB
from mlforge.cluster import DBSCAN, AgglomerativeClustering, GaussianMixture, KMeans
from mlforge.datasets import make_blobs, make_circles, make_moons
from mlforge.decomposition import PCA
from mlforge.neighbors import KNeighborsClassifier, KNeighborsRegressor
from mlforge.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler, train_test_split
from mlforge.svm import SVC
from mlforge.tree import DecisionTreeClassifier, DecisionTreeRegressor


@pytest.fixture(scope="module")
def reg():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6)
    y = X @ np.array([3, 0, -2, 0, 1.0, 0.5]) + 1 + rng.randn(200) * 0.4
    return X, y


@pytest.fixture(scope="module")
def clf():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y


# ------------------------------------------------------------- linear models
def test_ols_ridge_lasso_elasticnet(reg):
    X, y = reg
    pairs = [(L.LinearRegression(), sklm.LinearRegression(), 1e-10),
             (L.Ridge(alpha=2.5), sklm.Ridge(alpha=2.5), 1e-10),
             (L.Lasso(alpha=0.1, tol=1e-12, max_iter=5000), sklm.Lasso(alpha=0.1, tol=1e-12, max_iter=5000), 1e-6),
             (L.ElasticNet(alpha=0.1, l1_ratio=0.3, tol=1e-12, max_iter=5000),
              sklm.ElasticNet(alpha=0.1, l1_ratio=0.3, tol=1e-12, max_iter=5000), 1e-6)]
    for ours, ref, tol in pairs:
        ours.fit(X, y)
        ref.fit(X, y)
        assert np.allclose(ours.coef_, ref.coef_, atol=tol), type(ours).__name__
        assert ours.intercept_ == pytest.approx(ref.intercept_, abs=tol)
        assert np.allclose(ours.predict(X), ref.predict(X), atol=tol * 10)


def test_gradient_descent_solvers_reach_the_same_optimum(reg):
    X, y = reg
    gd = L.Ridge(alpha=2.5, solver="gd", learning_rate=0.1, n_iters=4000).fit(X, y)
    assert np.allclose(gd.coef_, sklm.Ridge(alpha=2.5).fit(X, y).coef_, atol=1e-6)


@pytest.mark.parametrize("C", [1.0, 0.05])
def test_regularised_logistic_newton_matches_lbfgs(clf, C):
    X, y = clf
    ours = L.LogisticRegression(solver="newton", C=C).fit(X, y)
    ref = sklm.LogisticRegression(C=C, tol=1e-12, max_iter=5000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_.ravel(), atol=1e-5)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=1e-5)
    assert np.abs(ours.predict_proba(X) - ref.predict_proba(X)).max() < 1e-6


def test_unregularised_logistic_matches_when_mle_exists():
    # overlapping classes => a finite maximum-likelihood solution exists
    # (on perfectly separable data the MLE diverges and no two solvers can agree).
    rng = np.random.RandomState(0)
    X = rng.randn(400, 4)
    y = (X @ np.array([1.5, -1.0, 0.5, 0.0]) + rng.logistic(size=400) > 0).astype(int)
    ours = L.LogisticRegression(solver="newton").fit(X, y)
    ref = sklm.LogisticRegression(C=1e12, tol=1e-12, max_iter=5000).fit(X, y)
    assert np.allclose(ours.coef_, ref.coef_.ravel(), atol=1e-4)
    assert np.abs(ours.predict_proba(X) - ref.predict_proba(X)).max() < 1e-5


def test_softmax_regression_matches_multinomial_logistic():
    X, y = load_wine(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    ours = L.SoftmaxRegression(C=1.0, learning_rate=0.5, n_iters=3000, optimizer="adam").fit(X, y)
    ref = sklm.LogisticRegression(C=1.0, tol=1e-12, max_iter=5000).fit(X, y)
    assert np.abs(ours.predict_proba(X) - ref.predict_proba(X)).max() < 5e-3
    assert (ours.predict(X) == ref.predict(X)).mean() > 0.99


# ------------------------------------------------------------------- trees
@pytest.mark.parametrize("loader", [load_iris, load_wine, load_breast_cancer])
@pytest.mark.parametrize("kw", [dict(), dict(max_depth=3), dict(ccp_alpha=0.01), dict(criterion="entropy", min_samples_leaf=3)])
def test_decision_tree_structure_matches(loader, kw):
    X, y = loader(return_X_y=True)
    ours = DecisionTreeClassifier(**kw).fit(X, y)
    ref = skt.DecisionTreeClassifier(random_state=0, **kw).fit(X, y)
    assert ours.get_n_leaves() == ref.get_n_leaves() and ours.get_depth() == ref.get_depth()
    assert (ours.predict(X) == ref.predict(X)).mean() >= 0.99


def test_decision_tree_regressor_matches():
    X, y = load_diabetes(return_X_y=True)
    ours = DecisionTreeRegressor(max_depth=4).fit(X, y)
    ref = skt.DecisionTreeRegressor(max_depth=4, random_state=0).fit(X, y)
    assert ours.get_n_leaves() == ref.get_n_leaves()
    assert np.allclose(ours.predict(X), ref.predict(X), atol=1e-8)
    # importances can differ only where two features give *identical* gain; the dominant ones agree
    top = np.argsort(ref.feature_importances_)[-2:]
    assert np.array_equal(np.argsort(ours.feature_importances_)[-2:], top)
    assert np.allclose(ours.feature_importances_[top], ref.feature_importances_[top], atol=1e-8)
    assert ours.feature_importances_.sum() == pytest.approx(1.0)


def test_forest_and_boosting_reach_comparable_accuracy():
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    from mlforge.tree import GradientBoostingClassifier, RandomForestClassifier
    ours_rf = RandomForestClassifier(100, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    ref_rf = ske.RandomForestClassifier(100, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours_rf - ref_rf) < 0.03
    ours_gb = GradientBoostingClassifier(100, random_state=0).fit(Xtr, ytr).score(Xte, yte)
    ref_gb = ske.GradientBoostingClassifier(random_state=0).fit(Xtr, ytr).score(Xte, yte)
    assert abs(ours_gb - ref_gb) < 0.03


def test_gradient_boosting_regressor_matches_closely():
    from mlforge.tree import GradientBoostingRegressor
    X, y = load_diabetes(return_X_y=True)
    ours = GradientBoostingRegressor(100, learning_rate=0.1, max_depth=3, random_state=0).fit(X[:350], y[:350])
    ref = ske.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X[:350], y[:350])
    assert abs(ours.score(X[350:], y[350:]) - ref.score(X[350:], y[350:])) < 0.08


# ---------------------------------------------------------- neighbours / bayes
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_knn_matches_on_tie_free_data(weights, metric):
    # continuous features => no distance ties, so the neighbour sets are unique
    rng = np.random.RandomState(0)
    X = rng.randn(250, 4)
    y = (X[:, 0] + X[:, 1] ** 2 > 0.5).astype(int) + (X[:, 2] > 1)
    tr, te = slice(0, 180), slice(180, None)
    ours = KNeighborsClassifier(5, metric=metric, weights=weights).fit(X[tr], y[tr])
    ref = skn.KNeighborsClassifier(5, metric=metric, weights=weights).fit(X[tr], y[tr])
    assert np.abs(ours.predict_proba(X[te]) - ref.predict_proba(X[te])).max() < 1e-9
    ours_r = KNeighborsRegressor(4, metric=metric, weights=weights).fit(X[tr], X[tr, 0] * 2)
    ref_r = skn.KNeighborsRegressor(4, metric=metric, weights=weights).fit(X[tr], X[tr, 0] * 2)
    assert np.allclose(ours_r.predict(X[te]), ref_r.predict(X[te]), atol=1e-9)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "chebyshev"])
def test_knn_neighbour_distances_match_even_with_ties(metric):
    # iris is full of duplicate / equidistant points: which tied neighbour is
    # returned is implementation defined, but the k smallest distances are not.
    X, y = load_iris(return_X_y=True)
    p = np.random.RandomState(0).permutation(150)
    tr, te = p[:100], p[100:]
    d_ours, _ = KNeighborsClassifier(7, metric=metric).fit(X[tr], y[tr]).kneighbors(X[te])
    d_ref, _ = skn.KNeighborsClassifier(7, metric=metric).fit(X[tr], y[tr]).kneighbors(X[te])
    assert np.allclose(d_ours, d_ref, atol=1e-12)


def test_naive_bayes_matches():
    X, y = load_iris(return_X_y=True)
    assert np.abs(GaussianNB().fit(X, y).predict_proba(X) - sknb.GaussianNB().fit(X, y).predict_proba(X)).max() < 1e-9
    C = np.random.RandomState(0).poisson(3, (120, 8))
    yc = (C[:, 0] + C[:, 1] > 6).astype(int)
    for a, b in [(MultinomialNB(0.5), sknb.MultinomialNB(alpha=0.5)), (BernoulliNB(0.5, 3), sknb.BernoulliNB(alpha=0.5, binarize=3))]:
        assert np.abs(a.fit(C, yc).predict_proba(C) - b.fit(C, yc).predict_proba(C)).max() < 1e-9


# --------------------------------------------------------------- unsupervised
def test_kmeans_matches_best_inertia():
    X, _ = make_blobs(400, 5, cluster_std=1.0, random_state=11)
    ours = KMeans(5, n_init=10, random_state=0).fit(X)
    ref = skc.KMeans(5, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-3)
    assert skm.adjusted_rand_score(ours.labels_, ref.labels_) > 0.99


@pytest.mark.parametrize("method", ["single", "complete", "average", "ward"])
def test_hierarchical_matches(method):
    X, _ = load_iris(return_X_y=True)
    ours = AgglomerativeClustering(3, method).fit(X)
    ref = skc.AgglomerativeClustering(3, linkage=method).fit(X)
    assert skm.adjusted_rand_score(ours.labels_, ref.labels_) == pytest.approx(1.0)


def test_dbscan_and_gmm_match():
    X, _ = make_moons(300, 0.06, random_state=0)
    assert skm.adjusted_rand_score(DBSCAN(0.2, 5).fit(X).labels_, skc.DBSCAN(eps=0.2, min_samples=5).fit(X).labels_) == 1.0
    Xg, _ = make_blobs(300, [[0, 0], [6, 6]], cluster_std=[1.0][0], random_state=0)
    ours = GaussianMixture(2, random_state=0).fit(Xg)
    ref = skmix.GaussianMixture(2, random_state=0).fit(Xg)
    assert ours.score(Xg) == pytest.approx(ref.score(Xg), abs=1e-4)
    assert ours.bic(Xg) == pytest.approx(ref.bic(Xg), abs=0.5)


@pytest.mark.parametrize("solver", ["svd", "eigh"])
def test_pca_matches(solver):
    X, _ = load_wine(return_X_y=True)
    ours, ref = PCA(4, solver=solver).fit(X), skd.PCA(4).fit(X)
    assert np.allclose(ours.explained_variance_, ref.explained_variance_, rtol=1e-9)
    assert np.allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, atol=1e-12)
    assert np.allclose(np.abs(ours.transform(X)), np.abs(ref.transform(X)), atol=1e-8)       # equal up to sign
    assert np.allclose(PCA(0.9).fit(X).n_components_, skd.PCA(0.9).fit(X).n_components_)


# --------------------------------------------------------------------- SVM
@pytest.mark.parametrize("name,data", [("moons", make_moons(250, 0.15, random_state=0)),
                                         ("circles", make_circles(250, 0.08, random_state=0))])
def test_kernel_svm_matches(name, data):
    X, y = data
    ours = SVC(C=1.0, kernel="rbf", random_state=0).fit(X[:180], y[:180])
    ref = sksvm.SVC(C=1.0, kernel="rbf").fit(X[:180], y[:180])
    assert (ours.predict(X[180:]) == ref.predict(X[180:])).mean() >= 0.97
    assert abs(len(ours.support_) - len(ref.support_)) <= max(3, 0.1 * len(ref.support_))


def test_linear_svm_multiclass_matches():
    X, y = make_blobs(300, [[-4, 0], [4, 0], [0, 5]], cluster_std=1.3, random_state=3)
    ours = SVC(kernel="linear", random_state=0).fit(X[:200], y[:200])
    ref = sksvm.SVC(kernel="linear").fit(X[:200], y[:200])
    assert (ours.predict(X[200:]) == ref.predict(X[200:])).mean() >= 0.97


# --------------------------------------------------------------- preprocessing
def test_scalers_and_encoders_match(reg):
    X, _ = reg
    for a, b in [(StandardScaler(), skp.StandardScaler()), (MinMaxScaler((-2, 3)), skp.MinMaxScaler((-2, 3))),
                 (RobustScaler(), skp.RobustScaler()), (PolynomialFeatures(3), skp.PolynomialFeatures(3))]:
        assert np.allclose(np.sort(a.fit_transform(X[:, :3]), axis=0), np.sort(b.fit_transform(X[:, :3]), axis=0), atol=1e-10), type(a)
    cats = np.array([["a", "x"], ["b", "y"], ["a", "z"]])
    assert np.array_equal(OneHotEncoder().fit_transform(cats), skp.OneHotEncoder().fit_transform(cats).toarray())


# ------------------------------------------------------------------- metrics
def test_all_metrics_match():
    rng = np.random.RandomState(0)
    yt = rng.randint(0, 3, 300)
    yp = np.where(rng.rand(300) < 0.7, yt, rng.randint(0, 3, 300))
    P = rng.dirichlet(np.ones(3), 300)
    for avg in ("macro", "micro", "weighted"):
        for name in ("precision_score", "recall_score", "f1_score"):
            assert getattr(M, name)(yt, yp, average=avg) == pytest.approx(getattr(skm, name)(yt, yp, average=avg))
    assert np.array_equal(M.confusion_matrix(yt, yp), skm.confusion_matrix(yt, yp))
    for name in ("balanced_accuracy_score", "cohen_kappa_score", "matthews_corrcoef", "accuracy_score"):
        assert getattr(M, name)(yt, yp) == pytest.approx(getattr(skm, name)(yt, yp))
    assert M.log_loss(yt, P) == pytest.approx(skm.log_loss(yt, P))
    assert M.roc_auc_score(yt, P) == pytest.approx(skm.roc_auc_score(yt, P, multi_class="ovr"))
    yb, sb = (yt == 1).astype(int), P[:, 1]
    assert M.roc_auc_score(yb, sb) == pytest.approx(skm.roc_auc_score(yb, sb))
    assert M.average_precision_score(yb, sb) == pytest.approx(skm.average_precision_score(yb, sb))
    assert M.brier_score_loss(yb, sb) == pytest.approx(skm.brier_score_loss(yb, sb))
    fpr, tpr, _ = M.roc_curve(yb, sb)
    rf, rt, _ = skm.roc_curve(yb, sb, drop_intermediate=False)
    assert M.auc(fpr, tpr) == pytest.approx(skm.auc(rf, rt))
    yr, pr = rng.randn(100), rng.randn(100)
    for name in ("mean_absolute_error", "mean_squared_error", "r2_score", "median_absolute_error", "max_error",
                 "explained_variance_score", "mean_absolute_percentage_error"):
        assert getattr(M, name)(yr, pr) == pytest.approx(getattr(skm, name)(yr, pr))
    Xc, lab = make_blobs(150, 3, random_state=0)
    assert M.silhouette_score(Xc, lab) == pytest.approx(skm.silhouette_score(Xc, lab))
    assert M.davies_bouldin_score(Xc, lab) == pytest.approx(skm.davies_bouldin_score(Xc, lab))
    assert M.calinski_harabasz_score(Xc, lab) == pytest.approx(skm.calinski_harabasz_score(Xc, lab))
    other = rng.randint(0, 3, 150)
    assert M.adjusted_rand_score(lab, other) == pytest.approx(skm.adjusted_rand_score(lab, other))
    assert M.normalized_mutual_info_score(lab, other) == pytest.approx(skm.normalized_mutual_info_score(lab, other))


def test_linkage_heights_match_scipy():
    hierarchy = pytest.importorskip("scipy.cluster.hierarchy")
    from mlforge.cluster import linkage
    X = np.random.RandomState(0).randn(120, 4)
    for method in ("single", "complete", "average", "ward"):
        ours, ref = linkage(X, method), hierarchy.linkage(X, method=method)
        assert np.allclose(np.sort(ours[:, 2]), np.sort(ref[:, 2]), atol=1e-10), method
        assert np.array_equal(ours[:, 3].astype(int).max(), ref[:, 3].astype(int).max())
