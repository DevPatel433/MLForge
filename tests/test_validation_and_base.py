import os

import numpy as np
import pytest

from mlforge import Pipeline
from mlforge.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from mlforge.neighbors import KNeighborsClassifier
from mlforge.preprocessing import StandardScaler
from mlforge.tree import DecisionTreeClassifier
from mlforge.utils import NotFittedError, check_array, check_X_y


def test_1d_input_gives_helpful_error():
    with pytest.raises(ValueError, match="reshape"):
        LinearRegression().fit(np.arange(5.0), np.arange(5.0))


def test_nan_and_inf_rejected():
    X = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="NaN"):
        LinearRegression().fit(X, [1, 2])
    with pytest.raises(ValueError):
        check_array(np.array([[np.inf, 1.0]]))


def test_inconsistent_lengths_and_bad_types():
    with pytest.raises(ValueError, match="inconsistent"):
        LinearRegression().fit(np.ones((4, 2)), np.ones(3))
    with pytest.raises(ValueError):
        check_array([["a", "b"]])
    with pytest.raises(ValueError, match="2D"):
        check_array(np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="empty"):
        check_array(np.empty((0, 3)))
    with pytest.raises(ValueError, match="1D"):
        check_X_y(np.ones((3, 2)), np.ones((3, 2)))


def test_not_fitted_error(blobs2):
    X, y = blobs2
    for est in (LinearRegression(), LogisticRegression(), DecisionTreeClassifier(), StandardScaler()):
        with pytest.raises(NotFittedError):
            (est.transform if isinstance(est, StandardScaler) else est.predict)(X)


def test_feature_count_mismatch_at_predict_is_an_error(blobs2):
    X, y = blobs2
    m = LogisticRegression().fit(X, y)
    with pytest.raises(Exception):
        m.predict(np.ones((3, 5)))


def test_get_set_params_clone_repr():
    m = Ridge(alpha=3.0, solver="gd", n_iters=10)
    p = m.get_params()
    assert p["alpha"] == 3.0 and p["solver"] == "gd"
    c = m.clone()
    assert c is not m and c.get_params() == p
    m.set_params(alpha=0.1, learning_rate=0.5)
    assert m.alpha == 0.1 and m.learning_rate == 0.5
    assert "Ridge(alpha=0.1" in repr(m)
    assert "l1_ratio" not in Lasso().get_params()


def test_clone_does_not_share_fitted_state(reg_data):
    X, y, _ = reg_data
    a = LinearRegression().fit(X, y)
    b = a.clone()
    with pytest.raises(NotFittedError):
        b.predict(X)


def test_save_and_load_roundtrip(tmp_path, reg_data):
    X, y, _ = reg_data
    m = Ridge(alpha=1.0).fit(X, y)
    path = m.save(str(tmp_path / "model.pkl"))
    loaded = Ridge.load(path)
    assert np.allclose(loaded.predict(X), m.predict(X))
    assert os.path.exists(path)


def test_random_state_makes_training_deterministic(blobs3):
    from mlforge.neural_network import MLPClassifier
    from mlforge.tree import RandomForestClassifier
    X, y = blobs3
    a = RandomForestClassifier(10, random_state=3).fit(X, y).predict_proba(X)
    b = RandomForestClassifier(10, random_state=3).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)
    m1 = MLPClassifier((8,), n_epochs=5, random_state=2).fit(X, y).predict_proba(X)
    m2 = MLPClassifier((8,), n_epochs=5, random_state=2).fit(X, y).predict_proba(X)
    assert np.allclose(m1, m2)


def test_pipeline_params_and_clone(blobs2):
    X, y = blobs2
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier(3))])
    pipe.set_params(knn__k=7)
    assert pipe.named_steps["knn"].k == 7
    assert pipe.fit(X, y).score(X, y) > 0.9
    assert "Pipeline" in repr(pipe)
    assert pipe.clone().steps[1][1].k == 7


def test_pipeline_full_api(blobs3):
    from mlforge.decomposition import PCA
    from mlforge.linear_model import SoftmaxRegression
    from mlforge.svm import LinearSVC
    X, y = blobs3
    # transformer-only pipeline (no final predictor) and fit without y
    tf = Pipeline([("sc", StandardScaler()), ("pca", PCA(2))]).fit(X)
    assert tf.transform(X).shape == (len(X), 2)
    # parameters, step replacement, classes_ and delegation
    pipe = Pipeline([("sc", StandardScaler()), ("clf", SoftmaxRegression(n_iters=200))])
    params = pipe.get_params()
    assert params["sc"] is pipe.named_steps["sc"] and "steps" in params
    pipe.set_params(clf=LinearSVC(n_iters=200))
    assert isinstance(pipe.named_steps["clf"], LinearSVC)
    pipe.set_params(steps=[("sc", StandardScaler()), ("clf", SoftmaxRegression(n_iters=200))])
    pipe.fit(X, y)
    assert list(pipe.classes_) == [0, 1, 2]
    assert pipe.decision_function(X).shape == (len(X), 3) and pipe.predict_proba(X).shape == (len(X), 3)
    assert pipe.score(X, y) > 0.9


def test_pipeline_handles_missing_values_with_an_imputer():
    from mlforge.preprocessing import SimpleImputer
    rng = np.random.RandomState(0)
    X = rng.randn(60, 3)
    y = (X[:, 0] > 0).astype(int)
    X[rng.rand(60, 3) < 0.1] = np.nan
    pipe = Pipeline([("imp", SimpleImputer("mean")), ("sc", StandardScaler()), ("clf", LogisticRegression(solver="newton"))])
    assert pipe.fit(X, y).score(X, y) > 0.8
    with pytest.raises(ValueError, match="NaN"):                 # without an imputer NaNs are rejected
        LogisticRegression().fit(X, y)


def test_validation_edge_cases():
    with pytest.raises(ValueError, match="numeric"):
        check_X_y(np.ones((3, 2)), np.array(["a", "b", "c"]), y_numeric=True)
    with pytest.raises(ValueError, match="scalar"):
        check_array(5.0)
    assert check_array([1, 2, 3], allow_1d=True).shape == (3,)
    assert check_X_y(np.ones((3, 2)), np.ones((3, 1)))[1].shape == (3,)             # column vector y is flattened
