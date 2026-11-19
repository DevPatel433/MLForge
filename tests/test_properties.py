"""Property-based tests (Hypothesis).

Instead of hand-picked examples, each test states a *law* that must hold for
every input in a family, and Hypothesis searches for counterexamples
(then shrinks them to a minimal failing case).
"""
import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from mlforge import metrics as M
from mlforge.autograd import Tensor
from mlforge.cluster import KMeans
from mlforge.decomposition import PCA
from mlforge.linear_model import LinearRegression, Ridge
from mlforge.model_selection import KFold, StratifiedKFold, train_test_split
from mlforge.neighbors import KNeighborsClassifier
from mlforge.optimization import Adam, GradientDescent, Momentum
from mlforge.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from mlforge.tree import DecisionTreeClassifier
from mlforge.utils import check_gradient, pairwise_distances
from mlforge.utils.math_utils import sigmoid, softmax

settings.register_profile("mlforge", max_examples=60, deadline=None, derandomize=True,
                          suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
settings.load_profile("mlforge")

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False, width=64)


def matrices(min_rows=2, max_rows=25, min_cols=1, max_cols=6):
    return hnp.arrays(np.float64, st.tuples(st.integers(min_rows, max_rows), st.integers(min_cols, max_cols)), elements=finite)


def distinct_rows(X):
    return len(np.unique(np.round(X, 9), axis=0)) == len(X)


labels = st.lists(st.integers(0, 3), min_size=4, max_size=40)


# ------------------------------------------------------------------ preprocessing
@given(matrices())
def test_standard_scaler_round_trip_and_moments(X):
    s = StandardScaler().fit(X)
    Z = s.transform(X)
    assert np.allclose(s.inverse_transform(Z), X, atol=1e-6 * (1 + np.abs(X).max()))
    assert np.all(np.abs(Z.mean(0)) < 1e-6)
    stds = Z.std(0)
    assert np.all((np.abs(stds - 1) < 1e-6) | (stds < 1e-6))        # unit variance, or constant column -> 0


@given(matrices())
def test_minmax_scaler_range_and_round_trip(X):
    s = MinMaxScaler().fit(X)
    Z = s.transform(X)
    assert Z.min() >= -1e-9 and Z.max() <= 1 + 1e-9
    assert np.allclose(s.inverse_transform(Z), X, atol=1e-6 * (1 + np.abs(X).max()))


@given(st.lists(st.text(alphabet="abcxyz", min_size=1, max_size=3), min_size=1, max_size=30))
def test_encoders_round_trip(values):
    y = np.array(values)
    le = LabelEncoder()
    assert np.array_equal(le.inverse_transform(le.fit_transform(y)), y)
    ohe = OneHotEncoder()
    Z = ohe.fit_transform(y)
    assert np.all(Z.sum(1) == 1) and np.array_equal(ohe.inverse_transform(Z).ravel(), y)


# ---------------------------------------------------------------------- splitting
@given(st.integers(5, 80), st.floats(0.1, 0.9), st.integers(0, 10_000))
def test_train_test_split_partitions_the_data(n, frac, seed):
    X = np.arange(n).reshape(-1, 1)
    assume(0 < int(np.ceil(n * frac)) < n)
    Xtr, Xte = train_test_split(X, test_size=frac, random_state=seed)
    assert sorted(np.concatenate([Xtr.ravel(), Xte.ravel()])) == list(range(n))


@given(st.integers(6, 60), st.integers(2, 6), st.booleans())
def test_kfold_folds_partition_and_balance(n, k, shuffle):
    assume(k <= n)
    folds = [te for _, te in KFold(k, shuffle=shuffle, random_state=0).split(np.arange(n))]
    assert sorted(np.concatenate(folds)) == list(range(n))
    assert max(map(len, folds)) - min(map(len, folds)) <= 1


@given(st.lists(st.integers(0, 2), min_size=12, max_size=60))
def test_stratified_kfold_class_balance(y):
    y = np.array(y)
    counts = np.bincount(y)
    assume(counts[counts > 0].min() >= 3)
    for _, te in StratifiedKFold(3, shuffle=True, random_state=0).split(y, y):
        for c in np.unique(y):
            assert abs((y[te] == c).sum() - counts[c] / 3) <= 1 + 1e-9


# ------------------------------------------------------------------------ numerics
@given(hnp.arrays(np.float64, st.tuples(st.integers(1, 6), st.integers(1, 8)), elements=finite), finite)
def test_softmax_is_a_distribution_and_shift_invariant(Z, c):
    P = softmax(Z)
    assert np.all(P >= 0) and np.allclose(P.sum(1), 1)
    assert np.allclose(softmax(Z + c), P, atol=1e-9)


@given(hnp.arrays(np.float64, st.integers(2, 30), elements=st.floats(-50, 50)))
def test_sigmoid_bounds_and_monotonic(z):
    s = sigmoid(np.sort(z))
    assert np.all((s >= 0) & (s <= 1)) and np.all(np.diff(s) >= -1e-15)


@given(matrices(2, 15, 1, 4), matrices(2, 15, 1, 4))
def test_pairwise_distances_are_metrics(A, B):
    assume(A.shape[1] == B.shape[1])
    scale = 1 + np.abs(np.concatenate([A, B])).max()
    D = pairwise_distances(A, B)
    assert D.shape == (len(A), len(B)) and np.all(D >= 0)
    ref = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    assert np.allclose(D, ref, atol=1e-6 * scale)
    S = pairwise_distances(A)
    assert np.allclose(S, S.T) and np.allclose(np.diag(S), 0)
    # triangle inequality d(i,k) <= d(i,j) + d(j,k)
    assert np.all(S[:, None, :] <= S[:, :, None] + S[None, :, :] + 1e-6 * scale)


# ---------------------------------------------------------------------- metrics
@given(labels, st.randoms(use_true_random=False))
def test_classification_metric_bounds_and_identities(y, rnd):
    y = np.array(y)
    p = np.array([c if rnd.random() < 0.6 else rnd.randint(0, 3) for c in y])
    assert 0 <= M.accuracy_score(y, p) <= 1
    cm = M.confusion_matrix(y, p)
    assert cm.sum() == len(y) and np.trace(cm) == M.accuracy_score(y, p, normalize=False)
    for avg in ("macro", "micro", "weighted"):
        for fn in (M.precision_score, M.recall_score, M.f1_score):
            assert -1e-12 <= fn(y, p, average=avg) <= 1 + 1e-12
    assert -1 - 1e-9 <= M.matthews_corrcoef(y, p) <= 1 + 1e-9
    assert M.f1_score(y, y, average="macro") == pytest.approx(1.0)
    assert M.accuracy_score(y, y) == 1.0


@given(st.lists(st.integers(0, 1), min_size=6, max_size=40), st.lists(st.floats(0, 1), min_size=6, max_size=40))
def test_auc_symmetry_and_bounds(y, s):
    n = min(len(y), len(s))
    y, s = np.array(y[:n]), np.array(s[:n])
    assume(0 < y.sum() < n and len(np.unique(s)) == n)               # both classes, no tied scores
    auc = M.roc_auc_score(y, s)
    assert 0 <= auc <= 1
    assert auc + M.roc_auc_score(y, -s) == pytest.approx(1.0)       # reversing the ranking flips the AUC
    assert M.average_precision_score(y, s) <= 1 + 1e-12


@given(hnp.arrays(np.float64, st.integers(3, 40), elements=finite), hnp.arrays(np.float64, st.integers(3, 40), elements=finite))
def test_regression_metric_relations(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    assert M.mean_absolute_error(a, b) <= M.root_mean_squared_error(a, b) + 1e-9      # MAE <= RMSE
    assert M.mean_squared_error(a, a) == 0 and M.r2_score(a, b) <= 1 + 1e-12
    assert M.max_error(a, b) >= M.mean_absolute_error(a, b) - 1e-9


@given(labels, st.permutations([0, 1, 2, 3]))
def test_adjusted_rand_is_permutation_invariant(y, perm):
    y = np.array(y)
    z = np.array(perm)[y]
    assert M.adjusted_rand_score(y, z) == pytest.approx(1.0)
    assert M.normalized_mutual_info_score(y, z) == pytest.approx(1.0)


# ------------------------------------------------------------------------- models
@given(matrices(8, 40, 1, 4), st.floats(0.5, 5), st.integers(0, 3))
def test_ols_residuals_orthogonal_and_scale_equivariant(X, a, seed):
    assume(np.linalg.matrix_rank(X - X.mean(0)) == X.shape[1] and X.std(0).min() > 1e-3)
    y = np.random.RandomState(seed).randn(len(X))
    m = LinearRegression().fit(X, y)
    resid = y - m.predict(X)
    scale = 1 + np.abs(X).max() * np.abs(resid).max() * len(X)
    assert np.abs(resid.sum()) < 1e-6 * scale                         # intercept => residuals sum to zero
    assert np.abs((X - X.mean(0)).T @ resid).max() < 1e-6 * scale     # normal equations
    m2 = LinearRegression().fit(X, a * y)
    assert np.allclose(m2.predict(X), a * m.predict(X), atol=1e-6 * (1 + np.abs(a * y).max()))


@given(matrices(10, 40, 2, 4), st.floats(0.1, 100), st.integers(0, 3))
def test_ridge_shrinks_coefficients(X, alpha, seed):
    assume(X.std(0).min() > 1e-3)
    y = np.random.RandomState(seed).randn(len(X))
    ols, ridge = LinearRegression().fit(X, y), Ridge(alpha=alpha).fit(X, y)
    assert np.linalg.norm(ridge.coef_) <= np.linalg.norm(ols.coef_) + 1e-8


@given(matrices(5, 30, 2, 5))
def test_pca_orthonormal_descending_and_lossless(X):
    assume(X.std(0).min() > 1e-3)
    p = PCA(None).fit(X)
    k = p.n_components_
    assert np.allclose(p.components_ @ p.components_.T, np.eye(k), atol=1e-8)
    assert np.all(np.diff(p.explained_variance_) <= 1e-8) and p.explained_variance_ratio_.sum() <= 1 + 1e-9
    assert np.allclose(p.inverse_transform(p.transform(X)), X, atol=1e-6 * (1 + np.abs(X).max()))


@given(matrices(6, 30, 1, 4), st.integers(2, 4), st.integers(0, 5))
def test_kmeans_invariants(X, k, seed):
    assume(len(np.unique(X, axis=0)) >= k)
    km = KMeans(k, n_init=2, random_state=seed).fit(X)
    assert km.inertia_ >= 0 and set(km.labels_) <= set(range(k))
    assert np.array_equal(km.predict(X), km.labels_)
    inertias = [((X[:, None] - c[None]) ** 2).sum(-1).min(1).sum() for c in km.history]
    assert all(a >= b - 1e-6 * (1 + a) for a, b in zip(inertias[:-1], inertias[1:]))     # Lloyd never increases inertia


@given(matrices(6, 30, 1, 4), st.integers(0, 5))
def test_unrestricted_tree_fits_distinct_points_and_probas_are_valid(X, seed):
    assume(distinct_rows(X))
    y = np.random.RandomState(seed).randint(0, 3, len(X))
    t = DecisionTreeClassifier().fit(X, y)
    assert np.array_equal(t.predict(X), y)                             # memorises any labelling of distinct points
    assert np.allclose(t.predict_proba(X).sum(1), 1) and set(t.predict(X + 0.123)) <= set(y)
    total = t.feature_importances_.sum()
    assert total == 0.0 or total == pytest.approx(1.0)


@given(matrices(5, 30, 1, 3), st.integers(0, 5))
def test_one_nearest_neighbour_reproduces_training_labels(X, seed):
    assume(distinct_rows(X))
    y = np.random.RandomState(seed).randint(0, 3, len(X))
    assert np.array_equal(KNeighborsClassifier(1).fit(X, y).predict(X), y)


# ----------------------------------------------------------------- optimisers / AD
@given(hnp.arrays(np.float64, st.integers(1, 8), elements=finite), st.floats(0.1, 10))
def test_gradient_descent_step_is_linear_and_adam_step_bounded(g, c):
    gd = GradientDescent(0.1)
    assert np.allclose(gd.step(c * g), c * GradientDescent(0.1).step(g))
    first = Adam(0.05).step(g)
    assert np.all(np.abs(first) <= 0.05 * (1 + 1e-6))                  # bias-corrected first step is ~lr
    assert np.allclose(Momentum(0.1, 0.0).step(g), 0.1 * g)            # beta = 0 reduces to plain GD


@given(hnp.arrays(np.float64, st.tuples(st.integers(1, 4), st.integers(1, 4)), elements=st.floats(0.2, 2.0)),
       hnp.arrays(np.float64, st.tuples(st.integers(1, 4), st.integers(1, 4)), elements=st.floats(0.2, 2.0)))
def test_autograd_matches_numerical_gradient_on_random_expressions(A, B):
    assume(A.shape == B.shape)

    def f(v):
        a, b = Tensor(v[:A.size].reshape(A.shape)), Tensor(v[A.size:].reshape(B.shape))
        return float(((a * b + a.tanh() - b.exp() * 0.1) ** 2).mean().data)

    def g(v):
        a, b = Tensor(v[:A.size].reshape(A.shape), True), Tensor(v[A.size:].reshape(B.shape), True)
        ((a * b + a.tanh() - b.exp() * 0.1) ** 2).mean().backward()
        return np.concatenate([a.grad.ravel(), b.grad.ravel()])
    ok, err = check_gradient(f, g, np.concatenate([A.ravel(), B.ravel()]), tol=1e-5)
    assert ok, err
