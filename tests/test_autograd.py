import numpy as np
import pytest

from mlforge.autograd import Tensor, no_grad
from mlforge.utils import check_gradient


def grad_check(build, *shapes, seed=0, tol=1e-5):
    """Compare autograd with finite differences for scalar output ``build(*tensors)``."""
    rng = np.random.RandomState(seed)
    arrays = [rng.rand(*s) + 0.5 for s in shapes]          # positive: safe for log / pow
    sizes = [a.size for a in arrays]

    def unpack(v):
        out, pos = [], 0
        for a, s in zip(arrays, sizes):
            out.append(v[pos:pos + s].reshape(a.shape))
            pos += s
        return out

    def f(v):
        return float(build(*[Tensor(a) for a in unpack(v)]).data)

    def g(v):
        ts = [Tensor(a, requires_grad=True) for a in unpack(v)]
        build(*ts).backward()
        return np.concatenate([t.grad.ravel() for t in ts])

    ok, err = check_gradient(f, g, np.concatenate([a.ravel() for a in arrays]), tol=tol)
    assert ok, err


def test_basic_arithmetic():
    grad_check(lambda a, b: ((a * b + a / b - b ** 2) * 3.0 - a).sum(), (3, 2), (3, 2))


def test_broadcasting_and_reductions():
    grad_check(lambda a, b: ((a + b) * (a - b)).sum(), (4, 3), (3,))
    grad_check(lambda a, b: (a * b).mean(axis=0).sum(), (4, 3), (1, 3))
    grad_check(lambda a: (a.sum(axis=1, keepdims=True) * a).sum(), (3, 4))


def test_matmul_variants():
    grad_check(lambda a, b: (a @ b).sum(), (3, 4), (4, 2))
    grad_check(lambda a, b: (a @ b).sum(), (3, 4), (4,))
    grad_check(lambda a, b: (a @ b).sum(), (4,), (4, 3))
    grad_check(lambda a, b: a @ b, (5,), (5,))


@pytest.mark.parametrize("fn", ["exp", "log", "relu", "tanh", "sigmoid"])
def test_elementwise(fn):
    grad_check(lambda a: getattr(a, fn)().sum(), (3, 3))


def test_log_softmax_and_reshape_transpose():
    grad_check(lambda a, w: (a.log_softmax(axis=1) * w).sum(), (4, 3), (4, 3))
    grad_check(lambda a: (a.reshape(2, 6).T * 2.0).sum(), (3, 4))


def test_reverse_operators_and_neg():
    t = Tensor([2.0, 4.0], requires_grad=True)
    y = (1.0 - t) + (8.0 / t) + 2.0 * t + (-t)
    y.sum().backward()
    assert np.allclose(t.grad, [-1 - 2 + 2 - 1, -1 - 0.5 + 2 - 1])


def test_gradient_accumulates_over_reuse():
    x = Tensor([3.0], requires_grad=True)
    (x * x * x).sum().backward()
    assert np.allclose(x.grad, [27.0])


def test_no_grad_and_requires_grad():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad
    z = Tensor([1.0]) * Tensor([2.0])
    assert not z.requires_grad
    x.grad = np.ones(2)
    x.zero_grad()
    assert x.grad is None
    assert "Tensor" in repr(x) and x.shape == (2,)


def test_trains_xor_with_autograd():
    rng = np.random.RandomState(0)
    X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])
    W1, b1 = Tensor(rng.randn(2, 8) * 0.8, True), Tensor(np.zeros(8), True)
    W2, b2 = Tensor(rng.randn(8, 2) * 0.8, True), Tensor(np.zeros(2), True)
    onehot = np.eye(2)[y]
    for _ in range(400):
        for p in (W1, b1, W2, b2):
            p.zero_grad()
        logits = (X @ W1 + b1).tanh() @ W2 + b2
        loss = -(logits.log_softmax(axis=1) * onehot).sum() * (1 / 4)
        loss.backward()
        for p in (W1, b1, W2, b2):
            p.data -= 0.3 * p.grad
    with no_grad():
        pred = ((X @ W1 + b1).tanh() @ W2 + b2).data.argmax(1)
    assert np.array_equal(pred, y)


# ---- tests added after mutation testing exposed gaps -------------------------------------------
def test_relu_gradient_on_negative_zero_and_positive_inputs():
    # grad_check() only draws positive values, which hid bugs in the negative branch of relu
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 3.0], requires_grad=True)
    (x.relu() * Tensor([1.0, 2.0, 3.0, 4.0, 5.0])).sum().backward()
    assert x.grad.tolist() == [0.0, 0.0, 0.0, 4.0, 5.0]                 # subgradient at exactly 0 is 0


@pytest.mark.parametrize("fn", ["exp", "tanh", "sigmoid", "relu"])
def test_elementwise_gradients_with_mixed_signs(fn):
    from mlforge.utils import check_gradient
    rng = np.random.RandomState(1)
    v = rng.randn(12) * 2
    v[np.abs(v) < 0.05] = 0.5                                           # stay away from relu's kink
    f = lambda a: float(getattr(Tensor(a), fn)().sum().data)

    def g(a):
        t = Tensor(a, requires_grad=True)
        getattr(t, fn)().sum().backward()
        return t.grad
    ok, err = check_gradient(f, g, v)
    assert ok, err
