"""Convolution, pooling and LSTM layers: forward values vs naive loops, gradient checks, learning."""
import numpy as np
import pytest

from mlforge.neural_network import LSTM, Conv2D, CrossEntropyLoss, Dense, Flatten, MaxPool2D, MSELoss, ReLU, Sequential, Tanh
from mlforge.utils import check_gradient


def naive_conv(x, W, b, stride, padding):
    N, C, H, Wd = x.shape
    F, _, kh, kw = W.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    Ho, Wo = (H + 2 * padding - kh) // stride + 1, (Wd + 2 * padding - kw) // stride + 1
    out = np.zeros((N, F, Ho, Wo))
    for n in range(N):
        for f in range(F):
            for i in range(Ho):
                for j in range(Wo):
                    patch = xp[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    out[n, f, i, j] = (patch * W[f]).sum() + b[f]
    return out


def naive_maxpool(x, k):
    N, C, H, W = x.shape
    out = np.zeros((N, C, H // k, W // k))
    for i in range(H // k):
        for j in range(W // k):
            out[:, :, i, j] = x[:, :, i * k:(i + 1) * k, j * k:(j + 1) * k].max(axis=(2, 3))
    return out


def net_gradcheck(net, X, Y, loss):
    f = lambda th: (net.set_flat_params(th), net.loss_and_grad(X, Y, loss)[0])[1]
    g = lambda th: (net.set_flat_params(th), net.loss_and_grad(X, Y, loss)[1])[1]
    return check_gradient(f, g, net.get_flat_params())


@pytest.mark.parametrize("stride,padding,kernel", [(1, 0, 3), (1, 1, 3), (2, 0, 3), (2, 1, 3), (1, 2, 5), (3, 1, 2)])
def test_conv_forward_matches_naive_loops(stride, padding, kernel):
    rng = np.random.RandomState(0)
    x = rng.randn(2, 3, 9, 8)
    layer = Conv2D(3, 4, kernel, stride, padding, random_state=1)
    layer.b = rng.randn(4)
    assert np.allclose(layer.forward(x), naive_conv(x, layer.W, layer.b, stride, padding))


@pytest.mark.parametrize("stride,padding", [(1, 0), (1, 1), (2, 0), (2, 1)])
def test_conv_gradients_numerically(stride, padding):
    rng = np.random.RandomState(1)
    X = rng.randn(3, 2, 7, 7)
    net = Sequential([Conv2D(2, 3, 3, stride, padding, random_state=0), Tanh(), Flatten()])
    out_dim = net.forward(X).shape[1]
    net.layers.append(Dense(out_dim, 4, "xavier", 1))
    ok, err = net_gradcheck(net, X, rng.randint(0, 4, 3), CrossEntropyLoss())
    assert ok, err


def test_input_gradient_of_conv_is_correct():
    rng = np.random.RandomState(2)
    layer = Conv2D(2, 3, 3, 2, 1, random_state=0)
    x, w = rng.randn(2, 2, 6, 6), None
    out = layer.forward(x)
    w = rng.randn(*out.shape)
    f = lambda a: float((layer.forward(a.reshape(x.shape)) * w).sum())
    g = lambda a: (layer.forward(a.reshape(x.shape)), layer.backward(w))[1].ravel()
    ok, err = check_gradient(f, g, x.ravel())
    assert ok, err


@pytest.mark.parametrize("shape,k", [((2, 3, 8, 8), 2), ((1, 2, 9, 7), 2), ((2, 1, 9, 9), 3)])
def test_maxpool_forward_matches_naive_and_drops_ragged_edges(shape, k):
    x = np.random.RandomState(3).randn(*shape)
    out = MaxPool2D(k).forward(x)
    assert out.shape == (shape[0], shape[1], shape[2] // k, shape[3] // k)
    assert np.allclose(out, naive_maxpool(x, k))


def test_maxpool_routes_gradient_to_exactly_one_input_per_window():
    x = np.zeros((1, 1, 4, 4))                                   # all ties: only the FIRST max may get the gradient
    pool = MaxPool2D(2)
    pool.forward(x)
    dx = pool.backward(np.ones((1, 1, 2, 2)))
    assert dx.sum() == 4 and (dx > 0).sum() == 4
    x2 = np.random.RandomState(4).randn(2, 2, 6, 6)
    pool.forward(x2)
    g = np.random.RandomState(5).randn(2, 2, 3, 3)
    dx2 = pool.backward(g)
    assert np.isclose(dx2.sum(), g.sum()) and np.allclose(np.sort(np.abs(dx2[dx2 != 0])), np.sort(np.abs(g.ravel())))


def test_whole_cnn_gradient_check():
    rng = np.random.RandomState(6)
    X = rng.randn(4, 1, 10, 10)
    net = Sequential([Conv2D(1, 3, 3, random_state=0), ReLU(), MaxPool2D(2), Conv2D(3, 4, 2, padding=1, random_state=1),
                      Tanh(), Flatten()])
    net.layers.append(Dense(net.forward(X).shape[1], 5, "xavier", 2))
    ok, err = net_gradcheck(net, X, rng.randint(0, 5, 4), CrossEntropyLoss())
    assert ok, err


def test_flatten_round_trip_and_invalid_init():
    flat = Flatten()
    x = np.random.RandomState(0).randn(3, 2, 4, 5)
    assert flat.forward(x).shape == (3, 40) and np.array_equal(flat.backward(flat.forward(x)), x)
    with pytest.raises(ValueError):
        Conv2D(1, 1, init="bogus")


def test_cnn_learns_to_find_which_half_of_the_image_is_bright():
    rng = np.random.RandomState(0)
    n = 160
    y = rng.randint(0, 2, n)
    X = rng.rand(n, 1, 8, 8) * 0.3
    X[y == 0, :, :, :4] += 0.7
    X[y == 1, :, :, 4:] += 0.7
    net = Sequential([Conv2D(1, 4, 3, random_state=0), ReLU(), MaxPool2D(2), Flatten(), Dense(36, 2, "he", 1)])
    h = net.fit(X, y, CrossEntropyLoss(), "adam", 0.02, 15, 32, random_state=0,
                metric=lambda out, Y: float(np.mean(out.argmax(1) == Y)))
    assert h["loss"][-1] < h["loss"][0] * 0.5 and h["metric"][-1] > 0.95


# ------------------------------------------------------------------------------------- LSTM
@pytest.mark.parametrize("return_sequences", [False, True])
def test_lstm_gradients_numerically(return_sequences):
    rng = np.random.RandomState(7)
    X = rng.randn(3, 5, 4)
    layers = [LSTM(4, 6, return_sequences, random_state=0)]
    if return_sequences:
        layers.append(Flatten())
        net = Sequential(layers)
        net.layers.append(Dense(net.forward(X).shape[1], 3, "xavier", 1))
    else:
        net = Sequential(layers + [Dense(6, 3, "xavier", 1)])
    ok, err = net_gradcheck(net, X, rng.randint(0, 3, 3), CrossEntropyLoss())
    assert ok, err


def test_lstm_input_gradient_and_shapes():
    rng = np.random.RandomState(8)
    lstm = LSTM(3, 4, return_sequences=True, random_state=0)
    x = rng.randn(2, 6, 3)
    out = lstm.forward(x)
    assert out.shape == (2, 6, 4) and LSTM(3, 4).forward(x).shape == (2, 4)
    w = rng.randn(*out.shape)
    f = lambda a: float((lstm.forward(a.reshape(x.shape)) * w).sum())
    g = lambda a: (lstm.forward(a.reshape(x.shape)), lstm.backward(w))[1].ravel()
    ok, err = check_gradient(f, g, x.ravel())
    assert ok, err


def test_lstm_matches_a_hand_written_single_step():
    lstm = LSTM(1, 1, random_state=0)
    lstm.Wx[:] = np.array([[0.5, -0.3, 0.8, 0.1]])
    lstm.Wh[:] = 0.0
    lstm.b[:] = [0.0, 1.0, 0.0, 0.0]
    x = np.array([[[2.0]]])
    z = np.array([[1.0, -0.6, 1.6, 0.2]]) + [0, 1.0, 0, 0]
    i, _f, g, o = 1 / (1 + np.exp(-z[0, 0])), 1 / (1 + np.exp(-z[0, 1])), np.tanh(z[0, 2]), 1 / (1 + np.exp(-z[0, 3]))
    c = i * g                                                         # c_0 = 0 so the forget gate is irrelevant at t=1
    assert lstm.forward(x)[0, 0] == pytest.approx(o * np.tanh(c))
    assert LSTM(2, 5).b[5:10].tolist() == [1.0] * 5                   # forget-gate bias initialised to 1


def test_lstm_solves_the_adding_problem_better_than_the_baseline():
    # Two inputs per step: a random value and a 0/1 marker; the target is the sum of the two marked values.
    # Predicting the mean scores MSE ~ 1/6; solving the task needs memory across time steps.
    rng = np.random.RandomState(0)
    n, T = 600, 8
    vals = rng.rand(n, T)
    marks = np.zeros((n, T))
    for r in range(n):
        marks[r, rng.choice(T, 2, replace=False)] = 1
    X, y = np.stack([vals, marks], axis=2), (vals * marks).sum(1)
    net = Sequential([LSTM(2, 24, random_state=0), Dense(24, 1, "xavier", 1)])
    net.fit(X[:500], y[:500], MSELoss(), "adam", 0.01, 60, 50, random_state=0)
    mse = float(np.mean((net.forward(X[500:]).ravel() - y[500:]) ** 2))
    baseline = float(np.mean((y[500:] - y[:500].mean()) ** 2))
    assert mse < 0.4 * baseline, (mse, baseline)


def test_model_summary_describes_conv_lstm_and_dense_layers_correctly():
    net = Sequential([Conv2D(1, 8, 3, random_state=0), ReLU(), MaxPool2D(2), Flatten(), Dense(200, 10, "he", 1)])
    text = net.summary()
    assert "1 -> 8 (3x3)" in text and "200 -> 10" in text and f"{8 * 9 + 8 + 2000 + 10}" in text.split("Total trainable parameters:")[1]
    assert "6 -> 4 (LSTM)" in Sequential([LSTM(6, 4)]).summary()


def test_conv_initialisation_scales_by_fan_in_and_fan_out():
    he = Conv2D(16, 32, 3, init="he", random_state=0).W.std()
    xavier = Conv2D(16, 32, 3, init="xavier", random_state=0).W.std()
    assert he == pytest.approx(np.sqrt(2.0 / (16 * 9)), rel=0.05)
    assert xavier == pytest.approx(np.sqrt(2.0 / (16 * 9 + 32 * 9)), rel=0.05)      # fan_out = out_channels * kh * kw
    assert Conv2D(2, 2, 3, init="normal", random_state=0).W.std() < 0.02
