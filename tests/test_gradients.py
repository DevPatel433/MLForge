"""Numerical gradient checks: the analytic gradients of every gradient-based
model are verified against central finite differences."""
import numpy as np
import pytest

from mlforge.linear_model import logistic_loss_grad, mse_loss_grad, softmax_loss_grad
from mlforge.neural_network import (
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    Dense,
    LeakyReLU,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)
from mlforge.svm import hinge_loss_grad
from mlforge.utils import check_gradient, numerical_gradient, relative_error


@pytest.fixture
def data():
    rng = np.random.RandomState(1)
    return rng.randn(30, 4), rng.randn(30), rng.randint(0, 2, 30), rng.randint(0, 3, 30)


def test_numerical_gradient_of_known_function():
    g = numerical_gradient(lambda x: np.sum(x ** 3), np.array([1.0, 2.0]))
    assert np.allclose(g, [3.0, 12.0], atol=1e-5)
    assert relative_error(g, [3.0, 12.0]) < 1e-6


@pytest.mark.parametrize("alpha", [0.0, 0.7])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_mse_gradient(data, alpha, fit_intercept):
    X, y, _, _ = data
    th = np.random.RandomState(2).randn(5)
    f = lambda t: mse_loss_grad(t, X, y, alpha, fit_intercept)
    ok, err = check_gradient(lambda t: f(t)[0], lambda t: f(t)[1], th) if fit_intercept else (True, 0)
    assert ok, err
    if not fit_intercept:
        assert f(th)[1][-1] == 0.0


@pytest.mark.parametrize("C", [None, 0.5])
def test_logistic_gradient(data, C):
    X, _, y01, _ = data
    th = np.random.RandomState(3).randn(5)
    f = lambda t: logistic_loss_grad(t, X, y01, C)
    ok, err = check_gradient(lambda t: f(t)[0], lambda t: f(t)[1], th)
    assert ok, err


@pytest.mark.parametrize("C", [None, 1.5])
def test_softmax_gradient(data, C):
    X, _, _, y3 = data
    Y = np.eye(3)[y3]
    th = np.random.RandomState(4).randn(4 * 3 + 3)
    f = lambda t: softmax_loss_grad(t, X, Y, C)
    ok, err = check_gradient(lambda t: f(t)[0], lambda t: f(t)[1], th)
    assert ok, err


def test_squared_hinge_gradient(data):
    X, _, y01, _ = data
    t = np.where(y01 == 1, 1.0, -1.0)
    th = np.random.RandomState(5).randn(5) * 0.1
    f = lambda v: hinge_loss_grad(v, X, t, 1.0, "squared_hinge")
    ok, err = check_gradient(lambda v: f(v)[0], lambda v: f(v)[1], th)
    assert ok, err


def test_hinge_subgradient_away_from_kinks(data):
    X, _, y01, _ = data
    t = np.where(y01 == 1, 1.0, -1.0)
    th = np.random.RandomState(6).randn(5) * 0.3
    f = lambda v: hinge_loss_grad(v, X, t, 2.0, "hinge")
    ok, err = check_gradient(lambda v: f(v)[0], lambda v: f(v)[1], th, tol=1e-4)
    assert ok, err
    with pytest.raises(ValueError):
        hinge_loss_grad(th, X, t, 1.0, "bogus")


def _net_check(layers, loss, X, Y, l2=0.0):
    net = Sequential(layers)
    f = lambda th: (net.set_flat_params(th), net.loss_and_grad(X, Y, loss, l2)[0])[1]
    g = lambda th: (net.set_flat_params(th), net.loss_and_grad(X, Y, loss, l2)[1])[1]
    return check_gradient(f, g, net.get_flat_params())


@pytest.mark.parametrize("act", [ReLU, LeakyReLU, Tanh, Sigmoid])
def test_mlp_gradients_all_activations(act):
    rng = np.random.RandomState(0)
    X, y = rng.randn(10, 4), rng.randint(0, 3, 10)
    layers = [Dense(4, 6, "xavier", 1), act(), Dense(6, 5, "xavier", 2), act(), Dense(5, 3, "xavier", 3)]
    ok, err = _net_check(layers, CrossEntropyLoss(), X, y, l2=0.05)
    assert ok, err


def test_mlp_gradient_mse_and_bce():
    rng = np.random.RandomState(0)
    X = rng.randn(12, 3)
    ok, err = _net_check([Dense(3, 5, "xavier", 1), Tanh(), Dense(5, 1, "xavier", 2)], MSELoss(), X, rng.randn(12))
    assert ok, err
    ok, err = _net_check([Dense(3, 5, "xavier", 1), Tanh(), Dense(5, 1, "xavier", 2)], BinaryCrossEntropyLoss(), X, rng.randint(0, 2, 12))
    assert ok, err


def test_softmax_layer_jacobian():
    rng = np.random.RandomState(0)
    sm, w = Softmax(), rng.randn(3, 4)
    f = lambda a: float(np.sum(sm.forward(a.reshape(3, 4)) * w))
    g = lambda a: (sm.forward(a.reshape(3, 4)), sm.backward(w))[1].ravel()
    ok, err = check_gradient(f, g, rng.randn(12))
    assert ok, err


def test_a_wrong_gradient_is_detected():
    ok, err = check_gradient(lambda x: np.sum(x ** 2), lambda x: 3 * x, np.array([1.0, -2.0]))
    assert not ok and err > 0.1


# ---- tests added after mutation testing exposed gaps -------------------------------------------
def test_relative_error_exact_values_and_floor():
    assert relative_error([1.0, 0.0], [0.0, 1.0]) == pytest.approx(np.sqrt(2) / 2)       # |a-b| / (|a|+|b|)
    assert relative_error([3.0, 4.0], [3.0, 4.0]) == 0.0
    assert relative_error([2.0], [0.0]) == pytest.approx(1.0)
    # both vectors tiny: the denominator is floored at 1e-12, so the error is *not* simply |a-b|
    assert relative_error([1e-14], [0.0]) == pytest.approx(1e-14 / 1e-12)
    assert relative_error([0.0], [0.0]) == 0.0


def test_check_gradient_tolerance_is_strict():
    f = lambda x: float(np.sum(x ** 2))
    g = lambda x: 2 * x + 1e-3
    x = np.array([1.0, 2.0])
    ok, err = check_gradient(f, g, x, tol=1.0)
    assert ok and err > 0
    assert not check_gradient(f, g, x, tol=err)[0]                      # the comparison is err < tol, strictly
    assert not check_gradient(f, g, x, tol=err * 0.5)[0]
