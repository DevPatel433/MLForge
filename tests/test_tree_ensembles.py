import numpy as np
import pytest

from mlforge.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)


def test_tree_memorises_training_data_when_unrestricted(blobs3):
    X, y = blobs3
    t = DecisionTreeClassifier().fit(X, y)
    assert t.score(X, y) == 1.0 and t.get_depth() >= 1 and t.node_count() >= 2 * t.get_n_leaves() - 1


def test_xor_needs_depth_two():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]] * 5, float)
    y = np.array([0, 1, 1, 0] * 5)
    assert DecisionTreeClassifier(max_depth=1).fit(X, y).score(X, y) < 1.0
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).score(X, y) == 1.0


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_criteria_work(blobs3, criterion):
    X, y = blobs3
    assert DecisionTreeClassifier(criterion=criterion, max_depth=4).fit(X, y).score(X, y) > 0.95


def test_invalid_criterion(blobs2):
    X, y = blobs2
    with pytest.raises(ValueError):
        DecisionTreeClassifier(criterion="bogus").fit(X, y)


def test_hyperparameters_limit_growth(blobs3):
    X, y = blobs3
    X = X + np.random.RandomState(0).randn(*X.shape) * 2
    full = DecisionTreeClassifier().fit(X, y)
    assert DecisionTreeClassifier(max_depth=2).fit(X, y).get_depth() <= 2
    assert DecisionTreeClassifier(min_samples_leaf=20).fit(X, y).get_n_leaves() < full.get_n_leaves()
    assert DecisionTreeClassifier(min_samples_split=100).fit(X, y).get_n_leaves() < full.get_n_leaves()
    assert DecisionTreeClassifier(min_impurity_decrease=0.02).fit(X, y).get_n_leaves() < full.get_n_leaves()


def test_cost_complexity_pruning(blobs3):
    X, y = blobs3
    X = X + np.random.RandomState(1).randn(*X.shape) * 2
    leaves = [DecisionTreeClassifier(ccp_alpha=a, random_state=0).fit(X, y).get_n_leaves() for a in (0, 0.005, 0.02, 0.1, 1.0)]
    assert leaves == sorted(leaves, reverse=True) and leaves[-1] == 1
    alphas = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
    assert alphas[0] == 0 and np.all(np.diff(alphas) >= 0)


def test_feature_importances_identify_informative_feature():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 4)
    y = (X[:, 2] > 0).astype(int)
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    assert t.feature_importances_.sum() == pytest.approx(1.0)
    assert t.feature_importances_.argmax() == 2 and t.feature_importances_[2] > 0.9


def test_predict_proba_and_labels(blobs3):
    X, y = blobs3
    labels = np.array(["a", "b", "c"])[y]
    t = DecisionTreeClassifier(max_depth=3).fit(X, labels)
    p = t.predict_proba(X)
    assert np.allclose(p.sum(1), 1) and set(t.predict(X)) <= {"a", "b", "c"}
    assert t.root is t.root_


def test_export_text(blobs2):
    X, y = blobs2
    txt = DecisionTreeClassifier(max_depth=2).fit(X, y).export_text(["f0", "f1"])
    assert "f0" in txt or "f1" in txt
    assert "class:" in txt
    r = DecisionTreeRegressor(max_depth=2).fit(X, X[:, 0]).export_text()
    assert "value:" in r


def test_single_class_and_constant_features():
    X = np.ones((10, 2))
    t = DecisionTreeClassifier().fit(X, np.zeros(10, int))
    assert t.get_n_leaves() == 1 and np.all(t.predict(X) == 0)
    t2 = DecisionTreeClassifier().fit(X, np.arange(10) % 2)
    assert t2.get_n_leaves() == 1


def test_max_features_options(blobs3):
    X, y = blobs3
    for mf in ("sqrt", "log2", 1, 0.5, None):
        DecisionTreeClassifier(max_features=mf, random_state=0).fit(X, y)


def test_regression_tree_piecewise_constant():
    x = np.linspace(0, 10, 200).reshape(-1, 1)
    y = np.where(x[:, 0] < 5, 1.0, 4.0)
    t = DecisionTreeRegressor(max_depth=1).fit(x, y)
    assert t.score(x, y) == 1.0 and t.get_n_leaves() == 2
    assert t.predict([[1.0], [9.0]]).tolist() == [1.0, 4.0]


def test_random_forest_classifier_oob_and_proba(blobs3):
    X, y = blobs3
    X = X + np.random.RandomState(0).randn(*X.shape)
    rf = RandomForestClassifier(40, oob_score=True, random_state=0).fit(X, y)
    assert rf.score(X, y) > 0.9 and 0.8 < rf.oob_score_ <= 1.0
    assert np.allclose(rf.predict_proba(X).sum(1), 1) and rf.feature_importances_.sum() == pytest.approx(1.0)
    assert len(rf.estimators_) == 40


def test_forest_beats_single_tree_on_noisy_data():
    rng = np.random.RandomState(0)
    X = rng.randn(400, 6)
    y = ((X[:, 0] + X[:, 1] + rng.randn(400) * 0.8) > 0).astype(int)
    tr, te = slice(0, 250), slice(250, None)
    tree = DecisionTreeClassifier(random_state=0).fit(X[tr], y[tr]).score(X[te], y[te])
    forest = RandomForestClassifier(60, random_state=0).fit(X[tr], y[tr]).score(X[te], y[te])
    assert forest > tree


def test_random_forest_regressor(reg_data):
    X, y, _ = reg_data
    rf = RandomForestRegressor(30, oob_score=True, random_state=0).fit(X, y)
    assert rf.score(X, y) > 0.9 and rf.oob_score_ > 0.5
    assert RandomForestRegressor(5, bootstrap=False, random_state=0).fit(X, y).predict(X).shape == y.shape


def test_gradient_boosting_regressor_improves_with_stages(reg_data):
    X, y, _ = reg_data
    gb = GradientBoostingRegressor(60, learning_rate=0.2, max_depth=2, random_state=0).fit(X, y)
    assert gb.train_loss_[-1] < gb.train_loss_[0] and gb.score(X, y) > 0.9
    staged = list(gb.staged_predict(X))
    assert len(staged) == 60 and np.allclose(staged[-1], gb.predict(X))
    sub = GradientBoostingRegressor(20, subsample=0.5, random_state=0).fit(X, y)
    assert sub.predict(X).shape == y.shape


@pytest.mark.parametrize("fixture", ["blobs2", "blobs3"])
def test_gradient_boosting_classifier(request, fixture):
    X, y = request.getfixturevalue(fixture)
    gb = GradientBoostingClassifier(40, learning_rate=0.2, max_depth=2, random_state=0).fit(X, y)
    assert gb.score(X, y) > 0.95
    assert np.allclose(gb.predict_proba(X).sum(1), 1)
    assert gb.train_loss_[-1] < gb.train_loss_[0]
    assert gb.feature_importances_.shape == (X.shape[1],)


def test_adjacent_floats_one_ulp_apart_can_be_split_regression():
    # Found by Hypothesis: the midpoint of two values 1 ulp apart rounds up to the larger one, so the
    # split sent *both* samples left and tree building recursed forever (RecursionError).
    lo = 1.0
    for _ in range(4):                                             # midpoint ties-to-even: depends on the pair's parity
        hi = np.nextafter(lo, 2.0)
        if (lo + hi) / 2.0 == hi:
            break
        lo = hi
    assert (lo + hi) / 2.0 == hi                                   # the premise: midpoint == upper value
    X = np.array([[lo], [hi]])
    t = DecisionTreeClassifier().fit(X, [0, 1])
    assert t.predict(X).tolist() == [0, 1] and t.get_n_leaves() == 2
    r = DecisionTreeRegressor().fit(X, [0.0, 5.0])
    assert r.predict(X).tolist() == [0.0, 5.0]


def test_cost_complexity_pruning_path_matches_scikit_learn():
    from sklearn.datasets import load_iris, load_wine
    from sklearn.tree import DecisionTreeClassifier as SkTree
    for loader in (load_iris, load_wine):
        X, y = loader(return_X_y=True)
        ours = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X, y)
        ref = SkTree(random_state=0).cost_complexity_pruning_path(X, y).ccp_alphas
        assert np.allclose(np.unique(np.round(ours, 12)), np.unique(np.round(ref, 12)), atol=1e-10)


def test_export_text_exact_structure_and_default_decimals():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    t = DecisionTreeClassifier().fit(X, [0, 0, 1, 1])
    assert t.export_text() == "|--- x[0] <= 1.500\n|   |--- class: 0\n|--- x[0] >  1.500\n|   |--- class: 1"
    assert t.export_text(decimals=1) == "|--- x[0] <= 1.5\n|   |--- class: 0\n|--- x[0] >  1.5\n|   |--- class: 1"
    deep = DecisionTreeClassifier().fit(np.array([[0.0], [1.0], [2.0], [3.0]]), [0, 1, 0, 1]).export_text()
    assert "|   |   |---" in deep                                      # two levels of nesting indent by one step each
