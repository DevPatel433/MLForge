import numpy as np
import pytest

from mlforge.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Perceptron,
    PolynomialRegression,
    Ridge,
    SoftmaxRegression,
)
from mlforge.optimization import Adam, CosineAnnealing, Momentum


def test_ols_recovers_coefficients_exactly(reg_data):
    X, y, w = reg_data
    m = LinearRegression().fit(X, y)
    assert np.allclose(m.coef_, w, atol=0.15)
    assert m.score(X, y) > 0.99
    assert np.allclose(m.weights, m.coef_) and m.bias == m.intercept_


def test_gradient_descent_converges_to_closed_form(reg_data):
    X, y, _ = reg_data
    exact = LinearRegression().fit(X, y)
    gd = LinearRegression(solver="gd", learning_rate=0.1, n_iters=2000).fit(X, y)
    assert np.allclose(gd.coef_, exact.coef_, atol=1e-5)
    assert gd.cost_history[0] > gd.cost_history[-1]
    assert np.all(np.diff(gd.cost_history_) <= 1e-12)


@pytest.mark.parametrize("opt", ["momentum", "adam", "rmsprop", Adam(0.05), Momentum(0.05)])
def test_every_optimizer_trains_linear_regression(reg_data, opt):
    X, y, _ = reg_data
    m = LinearRegression(solver="gd", optimizer=opt, learning_rate=0.03, n_iters=1500).fit(X, y)
    assert m.score(X, y) > 0.97


def test_minibatch_and_stochastic_gd(reg_data):
    X, y, _ = reg_data
    for bs in (16, 1):
        m = LinearRegression(solver="gd", optimizer="adam", learning_rate=0.02, n_iters=60,
                             batch_size=bs, random_state=0).fit(X, y)
        assert m.score(X, y) > 0.95


def test_early_stopping_with_tol(reg_data):
    X, y, _ = reg_data
    m = LinearRegression(solver="gd", learning_rate=0.1, n_iters=10000, tol=1e-10).fit(X, y)
    assert len(m.cost_history_) < 10000


def test_no_intercept(reg_data):
    X, y, _ = reg_data
    m = LinearRegression(fit_intercept=False).fit(X, y - y.mean())
    assert m.intercept_ == 0.0
    g = LinearRegression(fit_intercept=False, solver="gd", n_iters=500, learning_rate=0.1).fit(X, y)
    assert g.intercept_ == 0.0


def test_invalid_solver(reg_data):
    X, y, _ = reg_data
    with pytest.raises(ValueError):
        LinearRegression(solver="magic").fit(X, y)
    with pytest.raises(ValueError):
        ElasticNet(solver="magic").fit(X, y)
    with pytest.raises(ValueError):
        LogisticRegression(solver="magic").fit(X, (y > 0).astype(int))


def test_ridge_shrinks_and_gd_matches_closed_form(reg_data):
    X, y, _ = reg_data
    ols, ridge = LinearRegression().fit(X, y), Ridge(alpha=50.0).fit(X, y)
    assert np.linalg.norm(ridge.coef_) < np.linalg.norm(ols.coef_)
    gd = Ridge(alpha=5.0, solver="gd", learning_rate=0.1, n_iters=3000).fit(X, y)
    assert np.allclose(gd.coef_, Ridge(alpha=5.0).fit(X, y).coef_, atol=1e-5)


def test_lasso_is_sparse_and_solvers_agree(reg_data):
    X, y, w = reg_data
    cd = Lasso(alpha=0.2).fit(X, y)
    assert cd.coef_[1] == 0.0 and cd.coef_[3] == 0.0
    ista = Lasso(alpha=0.2, solver="ista", max_iter=20000, tol=1e-10).fit(X, y)
    assert np.allclose(cd.coef_, ista.coef_, atol=1e-4)
    assert cd.n_iter_ >= 1


def test_lasso_alpha_large_zeroes_everything(reg_data):
    X, y, _ = reg_data
    assert np.all(Lasso(alpha=100.0).fit(X, y).coef_ == 0)


def test_elasticnet_between_lasso_and_ridge(reg_data):
    X, y, _ = reg_data
    en = ElasticNet(alpha=0.2, l1_ratio=0.5).fit(X, y)
    assert en.predict(X).shape == y.shape
    assert ElasticNet(alpha=0.2, l1_ratio=1.0).fit(X, y).coef_ == pytest.approx(Lasso(alpha=0.2).fit(X, y).coef_)
    assert en.cost_history_[-1] <= en.cost_history_[0]


def test_constant_feature_does_not_break_cd():
    X = np.column_stack([np.ones(30), np.arange(30.0)])
    m = Lasso(alpha=0.01).fit(X, 2 * np.arange(30.0))
    assert m.coef_[0] == 0.0 and m.coef_[1] == pytest.approx(2.0, abs=0.05)


def test_polynomial_regression_fits_curve_and_1d_input():
    x = np.linspace(-2, 2, 60)
    y = 1 + x - 2 * x ** 2
    m = PolynomialRegression(degree=2).fit(x, y)
    assert m.score(x.reshape(-1, 1), y) > 0.9999
    lin = LinearRegression().fit(x.reshape(-1, 1), y)
    assert m.score(x, y) > lin.score(x.reshape(-1, 1), y)
    inter = PolynomialRegression(degree=2, interaction_only=True).fit(np.random.RandomState(0).randn(40, 2), np.zeros(40))
    assert inter.predict(np.zeros((1, 2))).shape == (1,)


def test_logistic_regression_basics(blobs2):
    X, y = blobs2
    m = LogisticRegression(n_iters=500).fit(X, y)
    assert m.score(X, y) > 0.95
    p = m.predict_proba(X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(1), 1)
    assert m.cost_history_[-1] < m.cost_history_[0]
    assert set(m.predict(X)) <= {0, 1}


def test_logistic_regression_preserves_arbitrary_labels(blobs2):
    X, y = blobs2
    labels = np.where(y == 1, "spam", "ham")
    m = LogisticRegression(solver="newton").fit(X, labels)
    assert set(m.predict(X)) == {"spam", "ham"} and list(m.classes_) == ["ham", "spam"]


def test_logistic_newton_and_gd_agree_with_regularisation(blobs2):
    X, y = blobs2
    Xn = X + np.random.RandomState(0).randn(*X.shape) * 1.5
    a = LogisticRegression(solver="newton", C=1.0).fit(Xn, y)
    b = LogisticRegression(solver="gd", C=1.0, learning_rate=0.5, n_iters=6000).fit(Xn, y)
    assert np.allclose(a.coef_, b.coef_, atol=1e-3)


def test_logistic_regularisation_shrinks_weights(blobs2):
    X, y = blobs2
    strong = LogisticRegression(solver="newton", C=0.001).fit(X, y)
    weak = LogisticRegression(solver="newton", C=100).fit(X, y)
    assert np.linalg.norm(strong.coef_) < np.linalg.norm(weak.coef_)


def test_logistic_threshold_and_decision_function(blobs2):
    X, y = blobs2
    m = LogisticRegression(solver="newton", threshold=0.99).fit(X, y)
    assert m.predict(X).sum() <= LogisticRegression(solver="newton").fit(X, y).predict(X).sum()
    assert m.decision_function(X).shape == (len(X),)


def test_logistic_rejects_multiclass(blobs3):
    X, y = blobs3
    with pytest.raises(ValueError, match="SoftmaxRegression"):
        LogisticRegression().fit(X, y)


def test_logistic_minibatch(blobs2):
    X, y = blobs2
    m = LogisticRegression(optimizer="adam", learning_rate=0.05, n_iters=30, batch_size=32, random_state=0).fit(X, y)
    assert m.score(X, y) > 0.95


def test_softmax_regression_multiclass(blobs3):
    X, y = blobs3
    m = SoftmaxRegression(n_iters=500).fit(X, y)
    assert m.score(X, y) > 0.95
    assert m.predict_proba(X).shape == (len(X), 3)
    reg = SoftmaxRegression(n_iters=500, C=0.01).fit(X, y)
    assert np.linalg.norm(reg.coef_) < np.linalg.norm(m.coef_)
    assert m.weights.shape == (2, 3)


def test_perceptron_converges_on_separable_data(blobs2, blobs3):
    X, y = blobs2
    p = Perceptron(random_state=0).fit(X, y)
    assert p.score(X, y) == 1.0 and p.errors_[-1] == 0
    X3, y3 = blobs3
    assert Perceptron(random_state=0).fit(X3, y3).score(X3, y3) > 0.9
    assert Perceptron(shuffle=False).fit(X, y).decision_function(X).shape == (len(X),)


def test_lr_schedule_improves_gd_stability(reg_data):
    X, y, _ = reg_data
    opt = Momentum(0.05, schedule=CosineAnnealing(500))
    m = LinearRegression(solver="gd", optimizer=opt, n_iters=500).fit(X, y)
    assert m.score(X, y) > 0.95


def test_elasticnet_ista_and_coordinate_descent_agree_with_a_real_l2_term(reg_data):
    # l1_ratio < 1 gives a non-zero ridge part: its gradient and the Lipschitz constant must be right in ISTA
    X, y, _ = reg_data
    for ratio in (0.2, 0.5, 0.8):
        cd = ElasticNet(alpha=0.3, l1_ratio=ratio, tol=1e-12, max_iter=20000).fit(X, y)
        ista = ElasticNet(alpha=0.3, l1_ratio=ratio, solver="ista", tol=1e-12, max_iter=50000).fit(X, y)
        assert np.allclose(cd.coef_, ista.coef_, atol=1e-5), ratio
        assert ista.cost_history_[-1] <= ista.cost_history_[0]
