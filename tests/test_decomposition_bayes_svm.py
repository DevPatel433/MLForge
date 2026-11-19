import numpy as np
import pytest

from mlforge.bayes import BernoulliNB, GaussianNB, MultinomialNB
from mlforge.datasets import make_circles
from mlforge.decomposition import PCA
from mlforge.svm import SVC, LinearSVC, kernel_matrix, smo


# ----------------------------------------------------------------------- PCA
@pytest.mark.parametrize("solver", ["svd", "eigh"])
def test_pca_properties(solver, rng):
    X = rng.randn(200, 5) @ rng.randn(5, 5)
    p = PCA(3, solver=solver).fit(X)
    assert p.components_.shape == (3, 5)
    assert np.allclose(p.components_ @ p.components_.T, np.eye(3), atol=1e-10)     # orthonormal
    assert np.all(np.diff(p.explained_variance_) <= 1e-12)
    Z = p.transform(X)
    assert np.allclose(Z.var(0, ddof=1), p.explained_variance_)
    assert np.allclose(np.cov(Z.T), np.diag(p.explained_variance_), atol=1e-8)        # decorrelated


def test_pca_full_rank_reconstruction_and_variance_target(rng):
    X = rng.randn(50, 4)
    p = PCA(None).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)), X) and p.reconstruction_error(X) < 1e-20
    assert p.explained_variance_ratio_.sum() == pytest.approx(1.0)
    X2 = np.column_stack([rng.randn(100) * 10, rng.randn(100) * 0.1])
    assert PCA(0.9).fit(X2).n_components_ == 1


def test_pca_whiten_and_errors(rng):
    X = rng.randn(100, 3) * [5, 2, 0.5]
    Z = PCA(3, whiten=True).fit_transform(X)
    assert np.allclose(Z.var(0, ddof=1), 1.0)
    p = PCA(2, whiten=True).fit(X)
    assert np.allclose(p.inverse_transform(p.transform(X)).shape, X.shape)
    with pytest.raises(ValueError):
        PCA(2, solver="x").fit(X)
    assert PCA(2).fit(X).components.shape == (3, 2)          # v0.1 alias


def test_pca_sign_is_deterministic(rng):
    X = rng.randn(60, 4)
    assert np.array_equal(PCA(2).fit(X).components_, PCA(2).fit(X.copy()).components_)


# --------------------------------------------------------------- Naive Bayes
def test_gaussian_nb(blobs3):
    X, y = blobs3
    m = GaussianNB().fit(X, y)
    assert m.score(X, y) > 0.95 and np.allclose(m.predict_proba(X).sum(1), 1)
    assert np.allclose(np.exp(m.predict_log_proba(X)).sum(1), 1) and np.allclose(m.class_prior_.sum(), 1)
    custom = GaussianNB(priors=[0.98, 0.01, 0.01]).fit(X, y)
    assert (custom.predict(X) == 0).sum() >= (m.predict(X) == 0).sum()


def test_multinomial_and_bernoulli_nb():
    rng = np.random.RandomState(0)
    X0 = rng.poisson([8, 1, 1], (60, 3))
    X1 = rng.poisson([1, 1, 8], (60, 3))
    X, y = np.vstack([X0, X1]), np.array([0] * 60 + [1] * 60)
    assert MultinomialNB().fit(X, y).score(X, y) > 0.9
    assert BernoulliNB(binarize=3).fit(X, y).score(X, y) > 0.9
    assert BernoulliNB(binarize=None).fit((X > 3).astype(float), y).score((X > 3).astype(float), y) > 0.9
    with pytest.raises(ValueError, match="non-negative"):
        MultinomialNB().fit(-X, y)


def test_multinomial_smoothing_handles_unseen_features():
    X = np.array([[3, 0], [4, 0], [0, 3], [0, 4]], float)
    y = np.array([0, 0, 1, 1])
    p = MultinomialNB(alpha=1.0).fit(X, y).predict_proba([[1, 1]])
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(), 1)


def test_naive_bayes_posterior_by_hand():
    # class 0 rows: x = [0, 0, 0, 1] -> P(x=1 | c0) = (1 + 1) / (4 + 2) = 1/3  (Laplace alpha = 1)
    # class 1 rows: x = [1, 1, 1, 0] -> P(x=1 | c1) = (3 + 1) / (4 + 2) = 2/3
    # equal priors => P(c1 | x=1) = (2/3) / (2/3 + 1/3) = 2/3
    X = np.array([[0], [0], [0], [1], [1], [1], [1], [0]], float)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    m = BernoulliNB(alpha=1.0, binarize=0.5).fit(X, y)
    assert m.predict_proba([[1.0]])[0, 1] == pytest.approx(2 / 3)
    assert m.predict_proba([[0.0]])[0, 1] == pytest.approx(1 / 3)


# ----------------------------------------------------------------------- SVM
def test_linear_svc_binary_and_multiclass(blobs2, blobs3):
    X, y = blobs2
    for loss in ("hinge", "squared_hinge"):
        assert LinearSVC(loss=loss, n_iters=400).fit(X, y).score(X, y) > 0.97
    X3, y3 = blobs3
    m = LinearSVC(n_iters=400).fit(X3, y3)
    assert m.score(X3, y3) > 0.9 and m.decision_function(X3).shape == (len(X3), 3)
    assert LinearSVC(n_iters=100, batch_size=32, random_state=0).fit(X, y).score(X, y) > 0.9


def test_linear_svc_C_controls_margin(blobs2):
    X, y = blobs2
    soft = LinearSVC(C=0.001, n_iters=600).fit(X, y)
    hard = LinearSVC(C=100, n_iters=600).fit(X, y)
    assert np.linalg.norm(soft.coef_) < np.linalg.norm(hard.coef_)
    assert hard.weights.shape == (2,) and isinstance(hard.bias, float)


def test_svc_rbf_solves_nonlinear_problems():
    X, y = make_circles(200, 0.06, random_state=0)
    lin = SVC(kernel="linear", random_state=0).fit(X, y)
    rbf = SVC(C=5, random_state=0).fit(X, y)
    assert rbf.score(X, y) > 0.97 > lin.score(X, y)
    assert 0 < len(rbf.support_) < len(X) and rbf.support_vectors_.shape[1] == 2
    assert rbf.n_support_.sum() >= len(rbf.support_) and rbf.dual_coef_.shape == (1, len(rbf.support_))


@pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
def test_all_kernels_train(kernel, blobs2):
    X, y = blobs2
    m = SVC(kernel=kernel, random_state=0).fit(X, y)
    assert m.score(X, y) > 0.8 and m.decision_function(X).shape == (len(X),)


def test_svc_dual_constraints_hold(blobs2):
    X, y = blobs2
    t = np.where(y == 1, 1.0, -1.0)
    K = kernel_matrix(X, X, "rbf", 0.5)
    alpha, b = smo(K, t, C=1.0, rng=np.random.RandomState(0))
    assert np.all(alpha >= -1e-9) and np.all(alpha <= 1 + 1e-9)
    assert abs(alpha @ t) < 1e-6                                   # sum a_i t_i = 0
    f = (alpha * t) @ K + b
    assert np.mean(np.sign(f) == t) > 0.95


def test_svc_multiclass_ovo_and_custom_kernel(blobs3):
    X, y = blobs3
    assert SVC(kernel="linear", random_state=0).fit(X, y).score(X, y) > 0.95
    custom = SVC(kernel=lambda A, B: A @ B.T, random_state=0).fit(X, y)
    assert custom.score(X, y) > 0.95
    assert SVC(gamma="auto", random_state=0).fit(X, y).gamma_ == pytest.approx(0.5)
    assert SVC(gamma=0.3, random_state=0).fit(X, y).gamma_ == 0.3
    with pytest.raises(ValueError):
        kernel_matrix(X, X, "bogus")


# ---- tests added after mutation testing exposed gaps -------------------------------------------
def test_pca_singular_values_and_whitened_inverse(rng):
    X = rng.randn(60, 4) * [4, 2, 1, 0.5]
    p = PCA(3).fit(X)
    assert np.allclose(p.singular_values_, np.linalg.svd(X - X.mean(0), compute_uv=False)[:3])
    full = PCA(None, whiten=True).fit(X)
    assert np.allclose(full.inverse_transform(full.transform(X)), X)            # whitening must be exactly undone
    assert np.allclose(full.transform(X).std(0, ddof=1), 1.0)


def test_multinomial_nb_default_laplace_smoothing_by_hand():
    # class 0 saw feature counts [3, 0], class 1 saw [0, 3]; alpha = 1 => theta_0 = [4/5, 1/5]
    X = np.array([[3, 0], [0, 3]], float)
    m = MultinomialNB().fit(X, [0, 1])
    assert m.alpha == 1.0
    assert np.allclose(np.exp(m.feature_log_prob_[0]), [4 / 5, 1 / 5])
    assert np.allclose(np.exp(MultinomialNB(alpha=2.0).fit(X, [0, 1]).feature_log_prob_[0]), [5 / 7, 2 / 7])   # [3+2, 0+2] / 7
