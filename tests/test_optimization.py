import numpy as np
import pytest

from mlforge.optimization import (
    AdaGrad,
    Adam,
    CosineAnnealing,
    ExponentialDecay,
    GradientDescent,
    InverseTimeDecay,
    Momentum,
    NesterovMomentum,
    RMSProp,
    StepDecay,
    WarmupSchedule,
    beale,
    beale_grad,
    get_optimizer,
    minimize,
    minimize_batches,
    quadratic_bowl,
    quadratic_bowl_grad,
    rosenbrock,
    rosenbrock_grad,
)
from mlforge.utils import check_gradient

LRS = {"gd": 0.1, "sgd": 0.1, "momentum": 0.05, "nesterov": 0.05, "adagrad": 0.8, "rmsprop": 0.05, "adam": 0.2}


@pytest.mark.parametrize("name", sorted(LRS))
def test_all_optimizers_minimise_a_quadratic(name):
    path = minimize(quadratic_bowl, quadratic_bowl_grad, [3.0, 2.0], name, 400, LRS[name])
    assert quadratic_bowl(path[-1]) < 1e-2 * quadratic_bowl(path[0])


def test_adam_and_momentum_solve_rosenbrock():
    for name, lr, steps in [("adam", 0.05, 3000), ("momentum", 0.0005, 3000)]:
        p = minimize(rosenbrock, rosenbrock_grad, [-1.0, 1.0], name, steps, lr)
        assert rosenbrock(p[-1]) < 1e-2


def test_test_function_gradients_are_correct():
    for f, g in [(rosenbrock, rosenbrock_grad), (beale, beale_grad), (quadratic_bowl, quadratic_bowl_grad)]:
        ok, err = check_gradient(f, g, np.array([0.7, 0.4]))
        assert ok, (f.__name__, err)


def test_momentum_accelerates_on_ill_conditioned_problem():
    gd = minimize(quadratic_bowl, quadratic_bowl_grad, [3, 2], "gd", 60, 0.01)
    mo = minimize(quadratic_bowl, quadratic_bowl_grad, [3, 2], "momentum", 60, 0.01)
    assert quadratic_bowl(mo[-1]) < quadratic_bowl(gd[-1])


def test_adam_first_step_is_learning_rate_sized():
    step = Adam(0.01).step(np.array([5.0, -0.001]))
    assert np.allclose(np.abs(step), 0.01, rtol=1e-3)


def test_reset_clears_state():
    opt = Momentum(0.1)
    opt.step(np.ones(2))
    assert opt.velocity is not None
    opt.reset()
    assert opt.velocity is None and opt.t == 0


def test_get_optimizer_variants_and_isolation():
    assert isinstance(get_optimizer("adam"), Adam)
    assert isinstance(get_optimizer(Momentum), Momentum)
    base = Adam(0.1)
    cp = get_optimizer(base)
    assert cp is not base
    assert isinstance(get_optimizer(None, 0.3), GradientDescent)
    assert "Adam" in repr(base)


def test_schedules():
    assert StepDecay(0.5, 10)(25) == 0.25
    assert ExponentialDecay(0.9)(2) == pytest.approx(0.81)
    assert InverseTimeDecay(1.0)(1) == 0.5
    c = CosineAnnealing(100, 0.1)
    assert c(0) == pytest.approx(1.0) and c(100) == pytest.approx(0.1) and c(500) == pytest.approx(0.1)
    assert WarmupSchedule(10)(0) == 0.1 and WarmupSchedule(10)(50) == 1.0


def test_schedule_is_applied_to_step():
    opt = GradientDescent(1.0, schedule=StepDecay(0.5, 1))
    assert np.allclose([opt.step(np.array([1.0]))[0] for _ in range(3)], [1.0, 0.5, 0.25])


def test_minimize_batches_minibatch_and_early_stopping():
    rng = np.random.RandomState(0)
    X, y = rng.randn(200, 3), None
    w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true

    def lg(theta, idx):
        Xb, yb = (X, y) if idx is None else (X[idx], y[idx])
        e = Xb @ theta - yb
        return np.mean(e ** 2), 2 * Xb.T @ e / len(yb)

    theta, hist = minimize_batches(lg, np.zeros(3), "adam", 200, 200, batch_size=20, learning_rate=0.05, random_state=0)
    assert np.allclose(theta, w_true, atol=0.05)
    theta2, hist2 = minimize_batches(lg, np.zeros(3), "gd", 5000, 200, tol=1e-12, learning_rate=0.1)
    assert len(hist2) < 5000 and np.allclose(theta2, w_true, atol=1e-3)


def test_minimize_batches_stops_on_divergence():
    f = lambda th, idx: (float(np.sum(th ** 2)), 2 * th)
    with np.errstate(over="ignore"):
        theta, hist = minimize_batches(f, np.ones(2), "gd", 200, learning_rate=50.0)
    assert len(hist) < 200


def test_optimizer_state_does_not_leak_between_fits(blobs2):
    from mlforge.linear_model import LogisticRegression
    X, y = blobs2
    opt = Adam(0.1)
    a = LogisticRegression(optimizer=opt, n_iters=50).fit(X, y).coef_
    b = LogisticRegression(optimizer=opt, n_iters=50).fit(X, y).coef_
    assert np.allclose(a, b)


# ---- tests added after mutation testing exposed gaps -------------------------------------------
@pytest.mark.parametrize("cls", [GradientDescent, Momentum, NesterovMomentum, AdaGrad, RMSProp, Adam])
def test_every_optimizer_counts_steps_and_applies_the_schedule(cls):
    opt = cls(learning_rate=0.1, schedule=StepDecay(0.5, 1))
    g = np.array([1.0])
    first = opt.step(g)
    assert opt.t == 1
    opt.step(g)
    opt.step(g)
    assert opt.t == 3                                         # one increment per step
    ref = cls(learning_rate=0.1)                              # same optimizer, no schedule
    assert opt.lr == 0.1 and ref.step(g) == pytest.approx(first)     # schedule(0) == 1: first step unscaled


def test_documented_default_hyperparameters():
    assert (GradientDescent().lr, Momentum().lr, NesterovMomentum().lr) == (0.01, 0.01, 0.01)
    assert (AdaGrad().lr, RMSProp().lr, Adam().lr) == (0.1, 0.01, 0.001)
    assert (Momentum().beta, NesterovMomentum().beta, RMSProp().beta) == (0.9, 0.9, 0.9)
    assert (Adam().beta1, Adam().beta2) == (0.9, 0.999)
    assert StepDecay()(99) == 1.0 and StepDecay()(100) == 0.5
    assert CosineAnnealing()(0) == pytest.approx(1.0) and CosineAnnealing()(500) == pytest.approx(0.5)
    assert CosineAnnealing()(1000) == pytest.approx(0.0) and WarmupSchedule()(24) == pytest.approx(0.5)


def test_exact_update_formulas_over_two_steps():
    g1, g2, lr = np.array([2.0, -1.0]), np.array([0.5, 3.0]), 0.1
    m = Momentum(lr, 0.9)                                      # Momentum: v = b v + lr g
    assert np.allclose(m.step(g1), lr * g1)
    assert np.allclose(m.step(g2), 0.9 * lr * g1 + lr * g2)
    n = NesterovMomentum(lr, 0.9)                              # Nesterov: v = b v + lr g ; update = b v + lr g
    v1 = lr * g1
    assert np.allclose(n.step(g1), 0.9 * v1 + lr * g1)
    v2 = 0.9 * v1 + lr * g2
    assert np.allclose(n.step(g2), 0.9 * v2 + lr * g2)
    a = AdaGrad(lr)                                            # AdaGrad: G += g^2 ; update = lr g / (sqrt(G) + eps)
    assert np.allclose(a.step(g1), lr * g1 / (np.abs(g1) + 1e-8))
    assert np.allclose(a.step(g2), lr * g2 / (np.sqrt(g1 ** 2 + g2 ** 2) + 1e-8))
    r = RMSProp(lr, 0.9)                                       # RMSProp: s = b s + (1-b) g^2 ; update = lr g / (sqrt(s) + eps)
    s1 = 0.1 * g1 ** 2
    assert np.allclose(r.step(g1), lr * g1 / (np.sqrt(s1) + 1e-8))
    s2 = 0.9 * s1 + 0.1 * g2 ** 2
    assert np.allclose(r.step(g2), lr * g2 / (np.sqrt(s2) + 1e-8))
    ad = Adam(lr, 0.9, 0.999)                                  # Adam with bias correction
    m1, v1a = 0.1 * g1, 0.001 * g1 ** 2
    assert np.allclose(ad.step(g1), lr * (m1 / 0.1) / (np.sqrt(v1a / 0.001) + 1e-8))
    m2, v2a = 0.9 * m1 + 0.1 * g2, 0.999 * v1a + 0.001 * g2 ** 2
    assert np.allclose(ad.step(g2), lr * (m2 / (1 - 0.9 ** 2)) / (np.sqrt(v2a / (1 - 0.999 ** 2)) + 1e-8))
