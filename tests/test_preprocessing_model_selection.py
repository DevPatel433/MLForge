import numpy as np
import pytest

from mlforge import Pipeline
from mlforge.linear_model import LinearRegression, LogisticRegression, Ridge
from mlforge.model_selection import (
    GridSearchCV,
    KFold,
    ParameterGrid,
    RandomizedSearchCV,
    ShuffleSplit,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    get_scorer,
    learning_curve,
    validation_curve,
)
from mlforge.neighbors import KNeighborsClassifier
from mlforge.preprocessing import (
    KNNImputer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
    train_test_split,
)


# --------------------------------------------------------------------- scalers
def test_standard_scaler(rng):
    X = rng.randn(100, 3) * [1, 10, 100] + [5, -5, 0]
    s = StandardScaler()
    Z = s.fit_transform(X)
    assert np.allclose(Z.mean(0), 0, atol=1e-12) and np.allclose(Z.std(0), 1)
    assert np.allclose(s.inverse_transform(Z), X)
    assert np.allclose(StandardScaler(with_mean=False).fit_transform(X).mean(0) != 0, True)
    const = StandardScaler().fit_transform(np.ones((5, 2)))
    assert np.all(const == 0)                                       # zero variance must not give NaN


def test_minmax_maxabs_robust_normalizer(rng):
    X = rng.randn(50, 3) * 4
    mm = MinMaxScaler((-1, 1))
    Z = mm.fit_transform(X)
    assert Z.min() == pytest.approx(-1) and Z.max() == pytest.approx(1)
    assert np.allclose(mm.inverse_transform(Z), X) and mm.min_.shape == (3,)
    ma = MaxAbsScaler().fit(X)
    assert np.abs(ma.transform(X)).max() == pytest.approx(1) and np.allclose(ma.inverse_transform(ma.transform(X)), X)
    Xo = np.vstack([X, [[1e6, 1e6, 1e6]]])
    r = RobustScaler().fit(Xo)
    assert np.all(np.abs(np.median(r.transform(Xo), axis=0)) < 1e-12) and np.allclose(r.inverse_transform(r.transform(Xo)), Xo)
    for norm, fn in [("l2", lambda v: np.linalg.norm(v, axis=1)), ("l1", lambda v: np.abs(v).sum(1)), ("max", lambda v: np.abs(v).max(1))]:
        assert np.allclose(fn(Normalizer(norm).fit_transform(X)), 1)
    with pytest.raises(ValueError):
        Normalizer("l3").fit_transform(X)
    assert np.all(Normalizer().fit_transform(np.zeros((2, 2))) == 0)


def test_polynomial_features():
    X = np.array([[2.0, 3.0]])
    full = PolynomialFeatures(2).fit_transform(X)
    assert sorted(full[0]) == sorted([1, 2, 3, 4, 6, 9])
    assert PolynomialFeatures(2, include_bias=False, interaction_only=True).fit_transform(X)[0].tolist() == [2, 3, 6]
    assert PolynomialFeatures(3, include_bias=False).fit(X).n_output_features_ == 9


# -------------------------------------------------------------------- encoders
def test_label_ordinal_onehot_encoders():
    y = np.array(["cat", "dog", "cat", "bird"])
    le = LabelEncoder()
    enc = le.fit_transform(y)
    assert enc.tolist() == [1, 2, 1, 0] and list(le.inverse_transform(enc)) == list(y)
    with pytest.raises(ValueError, match="unseen"):
        le.transform(["fish"])
    X = np.array([["a", "x"], ["b", "y"], ["a", "y"]])
    oe = OrdinalEncoder()
    assert oe.fit_transform(X).tolist() == [[0, 0], [1, 1], [0, 1]]
    assert (oe.inverse_transform(oe.transform(X)) == X).all()
    with pytest.raises(ValueError):
        oe.transform(np.array([["z", "x"]]))
    ohe = OneHotEncoder()
    Z = ohe.fit_transform(X)
    assert Z.shape == (3, 4) and Z.sum(1).tolist() == [2, 2, 2]
    assert (ohe.inverse_transform(Z) == X).all()
    assert ohe.get_feature_names(["c1", "c2"]) == ["c1=a", "c1=b", "c2=x", "c2=y"]
    with pytest.raises(ValueError):
        ohe.transform(np.array([["q", "x"]]))
    assert OneHotEncoder(handle_unknown="ignore").fit(X).transform(np.array([["q", "x"]])).tolist() == [[0, 0, 1, 0]]
    d = OneHotEncoder(drop_first=True).fit(X)
    assert d.transform(X).shape == (3, 2) and (d.inverse_transform(d.transform(X)) == X).all()
    assert OneHotEncoder().fit_transform(np.array(["a", "b", "a"])).shape == (3, 2)      # 1-D usage


# -------------------------------------------------------------------- imputers
def test_simple_imputer_strategies():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 4.0], [3.0, 8.0]])
    assert SimpleImputer("mean").fit_transform(X)[2, 0] == pytest.approx(7 / 3)
    assert SimpleImputer("median").fit_transform(X)[2, 0] == 3.0
    assert SimpleImputer("most_frequent").fit_transform(X)[0, 1] == 4.0
    assert SimpleImputer("constant", fill_value=-1).fit_transform(X)[0, 1] == -1
    out = SimpleImputer("mean", add_indicator=True).fit_transform(X)
    assert out.shape == (4, 4) and out[:, 2:].sum() == 2
    with pytest.raises(ValueError):
        SimpleImputer("bogus").fit(X)
    assert not np.isnan(SimpleImputer().fit(X).transform(X)).any()


def test_knn_imputer_uses_neighbours(rng):
    base = rng.randn(60, 1)
    X = np.hstack([base, 2 * base + rng.randn(60, 1) * 0.01])
    X_missing = X.copy()
    X_missing[5, 1] = np.nan
    filled = KNNImputer(3).fit_transform(X_missing)
    assert abs(filled[5, 1] - X[5, 1]) < 0.3 and not np.isnan(filled).any()
    all_nan = np.array([[np.nan, np.nan]])
    assert not np.isnan(KNNImputer().fit(X).transform(all_nan)).any()


# ------------------------------------------------------------------- splitting
def test_train_test_split_basic_and_errors():
    X, y = np.arange(40).reshape(20, 2), np.arange(20)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
    assert len(Xte) == 5 and len(Xtr) == 15 and set(ytr).isdisjoint(yte)
    assert np.array_equal(train_test_split(X, y, random_state=1)[0], train_test_split(X, y, random_state=1)[0])
    a, b = train_test_split(X, test_size=3, shuffle=False)
    assert np.array_equal(b, X[:3]) and len(a) == 17
    for bad in (0.0, 1.0, 25, 0):
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size=bad)
    with pytest.raises(ValueError):
        train_test_split(X, y[:5])


def test_stratified_split_keeps_proportions():
    y = np.array([0] * 90 + [1] * 10)
    X = np.arange(100).reshape(-1, 1)
    _, _, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    assert yte.mean() == pytest.approx(0.1, abs=0.02) and ytr.mean() == pytest.approx(0.1, abs=0.02)


def test_kfold_partitions_data():
    X = np.arange(23)
    seen = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        assert set(tr).isdisjoint(te) and len(tr) + len(te) == 23
        seen += list(te)
    assert sorted(seen) == list(range(23))
    assert KFold(5).get_n_splits() == 5
    with pytest.raises(ValueError):
        KFold(1)
    with pytest.raises(ValueError):
        list(KFold(30).split(X))


def test_stratified_kfold_and_shuffle_split():
    y = np.array([0] * 80 + [1] * 20)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(y, y):
        assert y[te].mean() == pytest.approx(0.2) and set(tr).isdisjoint(te)
    splits = list(ShuffleSplit(4, 0.25, random_state=0).split(y))
    assert len(splits) == 4 and all(len(te) == 25 for _, te in splits)


# ----------------------------------------------------------- CV and searching
def test_cross_val_score_and_predict(blobs2, reg_data):
    X, y = blobs2
    s = cross_val_score(LogisticRegression(solver="newton"), X, y, cv=5)
    assert s.shape == (5,) and s.mean() > 0.95
    assert cross_val_score(LogisticRegression(solver="newton"), X, y, cv=3, scoring="roc_auc").mean() > 0.98
    assert cross_val_score(LogisticRegression(solver="newton"), X, y, cv=3, scoring="neg_log_loss").mean() < 0
    Xr, yr, _ = reg_data
    assert cross_val_score(LinearRegression(), Xr, yr, cv=4, scoring="neg_mean_squared_error").mean() < 0
    assert cross_val_score(LinearRegression(), Xr, yr, cv=KFold(3), scoring=lambda e, a, b: e.score(a, b)).shape == (3,)
    pred = cross_val_predict(LogisticRegression(solver="newton"), X, y, cv=4)
    assert (pred == y).mean() > 0.95
    assert cross_val_predict(LinearRegression(), Xr, yr, cv=4).shape == yr.shape
    with pytest.raises(ValueError, match="Unknown scoring"):
        get_scorer("nope")


@pytest.mark.parametrize("scoring", ["accuracy", "balanced_accuracy", "f1", "f1_macro", "precision", "recall"])
def test_classification_scorers(blobs2, scoring):
    X, y = blobs2
    m = LogisticRegression(solver="newton").fit(X, y)
    assert get_scorer(scoring)(m, X, y) > 0.9


def test_parameter_grid():
    g = ParameterGrid({"a": [1, 2], "b": ["x", "y", "z"]})
    assert len(g) == 6 and {"a": 2, "b": "z"} in list(g)


def test_grid_search_selects_sensible_params(blobs3):
    X, y = blobs3
    gs = GridSearchCV(KNeighborsClassifier(), {"k": [1, 5, 15], "weights": ["uniform", "distance"]}, cv=4).fit(X, y)
    assert gs.best_score_ > 0.9 and gs.best_params_["k"] in (1, 5, 15)
    res = gs.cv_results_
    assert len(res["params"]) == 6 and res["rank_test_score"].min() == 1
    assert gs.best_score_ == pytest.approx(res["mean_test_score"].max())
    assert gs.predict(X).shape == y.shape and gs.score(X, y) > 0.9 and gs.predict_proba(X).shape == (len(X), 3)
    assert list(gs.classes_) == [0, 1, 2]


def test_grid_search_over_pipeline(blobs2):
    X, y = blobs2
    pipe = Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(solver="newton"))])
    gs = GridSearchCV(pipe, {"clf__C": [1e-4, 1.0]}, cv=3, scoring="neg_log_loss").fit(X, y)
    assert gs.best_params_["clf__C"] == 1.0


def test_grid_search_regression_alpha(reg_data):
    X, y, _ = reg_data
    gs = GridSearchCV(Ridge(), {"alpha": [1e-3, 1, 1e4]}, cv=4, scoring="neg_mean_squared_error").fit(X, y)
    assert gs.best_params_["alpha"] != 1e4


def test_randomized_search(blobs2):
    X, y = blobs2
    rs = RandomizedSearchCV(LogisticRegression(solver="newton"), {"C": lambda r: 10 ** r.uniform(-2, 2)},
                            n_iter=5, cv=3, random_state=0).fit(X, y)
    assert len(rs.cv_results_["params"]) == 5 and rs.best_score_ > 0.9
    rs2 = RandomizedSearchCV(KNeighborsClassifier(), {"k": [1, 3, 5]}, n_iter=3, cv=3, random_state=0).fit(X, y)
    assert rs2.best_params_["k"] in (1, 3, 5)


def test_learning_and_validation_curves(blobs2):
    X, y = blobs2
    sizes, tr, va = learning_curve(LogisticRegression(solver="newton"), X, y, train_sizes=[0.2, 0.5, 1.0], cv=4)
    assert tr.shape == va.shape == (3, 4) and np.all(np.diff(sizes) > 0)
    tr, va = validation_curve(KNeighborsClassifier(), X, y, "k", [1, 5, 25], cv=3)
    assert tr.shape == (3, 3) and tr[0].mean() == pytest.approx(1.0)       # k=1 memorises


def test_standard_scaler_constant_column_with_rounding_error_regression():
    # Found by Hypothesis: the mean of identical floats is off by ~1 ulp, so std() is ~1e-14 (not 0).
    # The scaler used to divide that noise by itself and return +-1 instead of 0.
    col = np.full((9, 1), 114.77580911)
    Z = StandardScaler().fit_transform(col)
    assert np.all(np.abs(Z) < 1e-9)
    mixed = np.column_stack([np.full(9, 0.1 + 0.2), np.arange(9.0)])           # 0.1+0.2 is inexact on purpose
    Z = StandardScaler().fit_transform(mixed)
    assert np.all(np.abs(Z[:, 0]) < 1e-9)
    assert Z[:, 1].std() == pytest.approx(1.0)


# ---- tests added after mutation testing exposed gaps -------------------------------------------
def test_one_hot_feature_names_with_drop_first():
    X = np.array([["a", "x"], ["b", "y"], ["a", "z"]])
    enc = OneHotEncoder(drop_first=True).fit(X)
    assert enc.get_feature_names(["c1", "c2"]) == ["c1=b", "c2=y", "c2=z"]       # first category of each column dropped
    assert enc.get_feature_names() == ["x0=b", "x1=y", "x1=z"]


def test_stratified_split_always_keeps_a_rare_class_in_the_test_set():
    y = np.array([0] * 40 + [1] * 2)                  # class 1 is so rare that rounding gives 0 test samples
    X = np.arange(42).reshape(-1, 1)
    _, _, _, yte = train_test_split(X, y, test_size=0.1, random_state=0, stratify=y)
    assert (yte == 1).sum() == 1                      # at least one, but not more than rounding up


def test_default_split_counts():
    assert len(list(KFold().split(np.arange(20)))) == 5
    assert KFold().get_n_splits() == 5 and ShuffleSplit().get_n_splits() == 10
    (tr, te), *_ = list(ShuffleSplit(random_state=0).split(np.arange(50)))
    assert len(te) == 10 and len(tr) == 40                               # default test_size = 0.2
