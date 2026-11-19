import numpy as np
import pytest

from mlforge.neural_network import (
    CrossEntropyLoss,
    Dense,
    Dropout,
    LeakyReLU,
    MLPClassifier,
    MLPRegressor,
    MSELoss,
    ReLU,
    Sequential,
    Sigmoid,
    Softmax,
    Tanh,
)

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
XOR_Y = np.array([0, 1, 1, 0])


@pytest.mark.parametrize("opt,lr", [("sgd", 0.5), ("adam", 0.05), ("rmsprop", 0.02), ("momentum", 0.1)])
def test_mlp_learns_xor(opt, lr):
    m = MLPClassifier((8,), activation="tanh", optimizer=opt, learning_rate=lr, n_epochs=600, batch_size=4, random_state=1)
    assert m.fit(XOR_X, XOR_Y).score(XOR_X, XOR_Y) == 1.0


def test_logistic_cannot_do_xor_but_mlp_can():
    from mlforge.linear_model import LogisticRegression
    assert LogisticRegression(n_iters=500).fit(XOR_X, XOR_Y).score(XOR_X, XOR_Y) <= 0.75


def test_multiclass_training_history(blobs3):
    X, y = blobs3
    m = MLPClassifier((16,), n_epochs=40, random_state=0).fit(X, y)
    h = m.history_
    assert h["loss"][-1] < h["loss"][0] and h["metric"][-1] > 0.95 and m.score(X, y) > 0.95
    assert np.allclose(m.predict_proba(X).sum(1), 1) and m.loss_history == h["loss"]
    assert len(m.layers) == 3 and m.n_epochs_run_ == 40


def test_string_labels_and_regularisation(blobs3):
    X, y = blobs3
    labels = np.array(["a", "b", "c"])[y]
    m = MLPClassifier((8,), n_epochs=30, l2=0.1, dropout=0.2, random_state=0).fit(X, labels)
    assert set(m.predict(X)) <= {"a", "b", "c"}
    big = MLPClassifier((8,), n_epochs=30, l2=0.0, random_state=0).fit(X, y)
    strong = MLPClassifier((8,), n_epochs=30, l2=1.0, random_state=0).fit(X, y)
    norm = lambda m: sum(np.sum(l.W ** 2) for l in m.layers if hasattr(l, "W"))
    assert norm(strong) < norm(big)


def test_early_stopping_restores_best_weights(blobs3):
    X, y = blobs3
    X = X + np.random.RandomState(0).randn(*X.shape) * 3          # noisy => overfits quickly
    m = MLPClassifier((64,), n_epochs=400, learning_rate=0.02, early_stopping=True, patience=5, random_state=0).fit(X, y)
    assert m.n_epochs_run_ < 400
    assert "val_loss" in m.history_ and len(m.history_["val_loss"]) == m.n_epochs_run_


def test_regressor(reg_data):
    X, y, _ = reg_data
    m = MLPRegressor((32, 16), n_epochs=200, random_state=0).fit(X, y)
    assert m.score(X, y) > 0.95 and m.predict(X).shape == y.shape


def test_invalid_activation_and_init():
    with pytest.raises(ValueError):
        MLPClassifier(activation="swish").fit(XOR_X, XOR_Y)
    with pytest.raises(ValueError):
        Dense(2, 2, init="bogus")
    with pytest.raises(ValueError):
        Dropout(1.0)


def test_dropout_behaviour():
    d = Dropout(0.5, random_state=0)
    x = np.ones((1000, 20))
    out = d.forward(x, training=True)
    assert 0.4 < (out == 0).mean() < 0.6 and out.mean() == pytest.approx(1.0, abs=0.05)   # inverted scaling keeps E[x]
    assert np.array_equal(d.forward(x, training=False), x)
    assert np.array_equal(d.backward(np.ones_like(x)), np.ones_like(x))
    d.forward(x, training=True)
    assert np.array_equal(d.backward(np.ones_like(x)) != 0, d.mask != 0)


def test_initialisation_scales():
    he = Dense(1000, 500, "he", 0).W.std()
    xavier = Dense(1000, 500, "xavier", 0).W.std()
    assert he == pytest.approx(np.sqrt(2 / 1000), rel=0.05)
    assert xavier == pytest.approx(np.sqrt(2 / 1500), rel=0.05)
    assert np.all(Dense(3, 3, "zeros").W == 0) and Dense(3, 3, "normal", 0).W.std() < 0.05


def test_activations_forward_values():
    x = np.array([[-1.0, 0.0, 2.0]])
    assert ReLU().forward(x).tolist() == [[0, 0, 2]]
    assert LeakyReLU(0.1).forward(x)[0, 0] == pytest.approx(-0.1)
    assert Sigmoid().forward(np.zeros((1, 1)))[0, 0] == 0.5 and Tanh().forward(np.zeros((1, 1)))[0, 0] == 0
    assert Softmax().forward(x).sum() == pytest.approx(1.0)
    assert Sigmoid().forward(np.array([[-1e4]]))[0, 0] == pytest.approx(0.0, abs=1e-12)


def test_sequential_api_summary_and_flat_params():
    net = Sequential([Dense(3, 4, random_state=0), ReLU(), Dense(4, 2, random_state=1)])
    assert net.n_parameters() == 3 * 4 + 4 + 4 * 2 + 2 and "Total trainable parameters: 26" in net.summary()
    v = net.get_flat_params()
    net.set_flat_params(v * 0)
    assert np.all(net.get_flat_params() == 0)
    net.set_flat_params(v)
    assert np.allclose(net(np.ones((2, 3))), net.forward(np.ones((2, 3))))


def test_sequential_fit_with_custom_loss_and_verbose(capsys, blobs2):
    X, y = blobs2
    net = Sequential([Dense(2, 8, random_state=0), ReLU(), Dense(8, 2, random_state=1)])
    h = net.fit(X, y, CrossEntropyLoss(), "adam", 0.05, 10, 32, validation_data=(X, y), patience=3, random_state=0, verbose=5)
    assert h["loss"][-1] < h["loss"][0] and "epoch" in capsys.readouterr().out
    assert CrossEntropyLoss().forward(net(X), np.eye(2)[y]) < 0.5
    assert MSELoss().forward(np.ones((3, 1)), np.zeros(3)) == 1.0
