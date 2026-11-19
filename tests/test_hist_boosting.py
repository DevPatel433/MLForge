import numpy as np
import pytest

from mlforge.datasets import make_blobs
from mlforge.tree import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from mlforge.tree.hist_boosting import BinMapper, HistTree, best_split_from_histograms, flat_bin_index, histograms


@pytest.fixture
def reg():
    rng = np.random.RandomState(0)
    X = rng.randn(500, 6)
    y = 3 * X[:, 0] + np.sin(2 * X[:, 1]) + rng.randn(500) * 0.3
    return X, y


def test_binning_is_monotone_ordered_and_consistent_with_thresholds():
    rng = np.random.RandomState(0)
    X = np.column_stack([rng.randn(1000), rng.randint(0, 5, 1000).astype(float), rng.exponential(size=1000)])
    bm = BinMapper(32).fit(X)
    Xb = bm.transform(X)
    assert Xb.dtype == np.uint8 and Xb.max() < 32
    for j, th in enumerate(bm.thresholds_):
        assert np.all(np.diff(th) > 0)                                   # strictly increasing thresholds
        order = np.argsort(X[:, j], kind="stable")
        assert np.all(np.diff(Xb[order, j].astype(int)) >= 0)             # bin id is monotone in the value
        b = Xb[:, j].astype(int)
        below = b < len(th)
        assert np.all(X[below, j] <= th[b[below]])                        # "value <= th[bin]"
        assert np.all(X[b > 0, j] > th[b[b > 0] - 1])                     # and strictly above the previous threshold
    assert len(bm.thresholds_[1]) == 4 and bm.n_bins_ == 32               # 5 distinct values -> 4 thresholds


def test_binning_separates_values_one_ulp_apart():
    lo = 1.0
    for _ in range(4):                       # find a pair whose midpoint rounds up to the larger value
        hi = np.nextafter(lo, 2.0)
        if (lo + hi) / 2.0 == hi:
            break
        lo = hi
    Xb = BinMapper().fit(np.array([[lo], [hi]])).transform(np.array([[lo], [hi]]))
    assert Xb[0, 0] != Xb[1, 0]


def test_invalid_max_bins():
    for bad in (1, 300):
        with pytest.raises(ValueError):
            BinMapper(bad)


def test_histogram_split_equals_exhaustive_search_on_binned_data():
    rng = np.random.RandomState(1)
    X = rng.randn(300, 5)
    g = rng.randn(300)
    h = rng.rand(300) + 0.5
    bm = BinMapper(16).fit(X)
    Xb = bm.transform(X)
    nb = bm.n_bins_
    hist = histograms(flat_bin_index(Xb, nb), g, h, 5, nb)
    G, H = g.sum(), h.sum()
    lam, msl = 0.7, 5
    f, b, gain = best_split_from_histograms(*hist, G, H, 300.0, lam, msl)
    best = (-np.inf, None, None)                                         # brute force: every (feature, bin) split
    for ff in range(5):
        for bb in range(nb - 1):
            left = Xb[:, ff] <= bb
            if left.sum() < msl or (~left).sum() < msl:
                continue
            gl, hl, gr, hr = g[left].sum(), h[left].sum(), g[~left].sum(), h[~left].sum()
            val = gl ** 2 / (hl + lam) + gr ** 2 / (hr + lam) - G ** 2 / (H + lam)
            if val > best[0]:
                best = (val, ff, bb)
    assert (f, b) == (best[1], best[2]) and gain == pytest.approx(best[0], rel=1e-6)


def test_histogram_subtraction_trick_is_exact():
    rng = np.random.RandomState(2)
    X = rng.randn(200, 4)
    g, h = rng.randn(200), rng.rand(200)
    Xb = BinMapper(20).fit(X).transform(X)
    flat = flat_bin_index(Xb, 20)
    parent = histograms(flat, g, h, 4, 20)
    left = np.arange(200) < 70
    small = histograms(flat[left], g[left], h[left], 4, 20)
    other = tuple(p - s for p, s in zip(parent, small))
    direct = histograms(flat[~left], g[~left], h[~left], 4, 20)
    for a, b in zip(other, direct):
        assert np.allclose(a, b)


def test_tree_respects_its_size_limits(reg):
    X, y = reg
    Xb = BinMapper().fit(X).transform(X)
    for leaves in (2, 5, 31):
        t = HistTree(max_leaf_nodes=leaves, min_samples_leaf=5).fit(Xb, -y, np.ones(len(y)))
        assert 2 <= t.n_leaves <= leaves
    t = HistTree(max_leaf_nodes=64, max_depth=2, min_samples_leaf=1).fit(Xb, -y, np.ones(len(y)))
    assert t.n_leaves <= 4
    big = HistTree(max_leaf_nodes=64, min_samples_leaf=100).fit(Xb, -y, np.ones(len(y)))
    sizes = np.bincount(np.searchsorted(np.unique(big.predict(Xb)), big.predict(Xb)))
    assert sizes.min() >= 100                                              # no leaf smaller than min_samples_leaf


def test_l2_regularisation_shrinks_leaf_values(reg):
    X, y = reg
    Xb = BinMapper().fit(X).transform(X)
    plain = HistTree(l2=0.0).fit(Xb, -y, np.ones(len(y)))
    shrunk = HistTree(l2=500.0).fit(Xb, -y, np.ones(len(y)))
    assert np.abs(shrunk.predict(Xb)).max() < np.abs(plain.predict(Xb)).max()


def test_regressor_learns_and_loss_decreases(reg):
    X, y = reg
    m = HistGradientBoostingRegressor(60, learning_rate=0.2, min_samples_leaf=10).fit(X[:400], y[:400])
    assert m.score(X[400:], y[400:]) > 0.9
    assert np.all(np.diff(m.train_loss_) <= 1e-9)                          # training loss never increases
    staged = list(m.staged_predict(X[400:]))
    assert len(staged) == 60 and np.allclose(staged[-1], m.predict(X[400:]))
    assert m.feature_importances_.sum() == pytest.approx(1.0) and m.feature_importances_.argmax() == 0


def test_regressor_matches_scikit_learn_quality(reg):
    sk = pytest.importorskip("sklearn.ensemble")
    X, y = reg
    ours = HistGradientBoostingRegressor(100, min_samples_leaf=20).fit(X[:400], y[:400])
    ref = sk.HistGradientBoostingRegressor(max_iter=100, early_stopping=False, random_state=0).fit(X[:400], y[:400])
    assert abs(ours.score(X[400:], y[400:]) - ref.score(X[400:], y[400:])) < 0.03


@pytest.mark.parametrize("n_classes", [2, 3])
def test_classifier(n_classes):
    centers = [[-3, 0], [3, 0], [0, 4]][:n_classes]
    X, y = make_blobs(300, centers, cluster_std=1.2, random_state=0)
    m = HistGradientBoostingClassifier(30, learning_rate=0.2, min_samples_leaf=10).fit(X[:200], y[:200])
    assert m.score(X[200:], y[200:]) > 0.9
    p = m.predict_proba(X)
    assert p.shape == (300, n_classes) and np.allclose(p.sum(1), 1) and np.all(p >= 0)
    assert np.all(np.diff(m.train_loss_) <= 1e-9)
    assert set(m.predict(X)) <= set(m.classes_) and m.decision_function(X).shape[0] == 300


def test_classifier_matches_scikit_learn_accuracy_and_is_deterministic():
    sk = pytest.importorskip("sklearn.ensemble")
    rng = np.random.RandomState(0)
    X = rng.randn(600, 8)
    y = (X[:, 0] + X[:, 1] ** 2 + rng.randn(600) * 0.5 > 1).astype(int)
    a = HistGradientBoostingClassifier(60).fit(X[:400], y[:400])
    b = sk.HistGradientBoostingClassifier(max_iter=60, early_stopping=False, random_state=0).fit(X[:400], y[:400])
    assert abs(a.score(X[400:], y[400:]) - b.score(X[400:], y[400:])) < 0.04
    again = HistGradientBoostingClassifier(60).fit(X[:400], y[:400])
    assert np.array_equal(a.predict_proba(X), again.predict_proba(X))


def test_string_labels_constant_target_and_get_params(reg):
    X, y = reg
    m = HistGradientBoostingClassifier(10, min_samples_leaf=10).fit(X, np.where(y > 0, "pos", "neg"))
    assert set(m.predict(X)) <= {"pos", "neg"}
    const = HistGradientBoostingRegressor(5).fit(X, np.full(len(X), 3.0))
    assert np.allclose(const.predict(X), 3.0)                              # no gain anywhere -> only the baseline
    assert HistGradientBoostingRegressor(max_leaf_nodes=7).clone().max_leaf_nodes == 7


def test_max_depth_is_counted_from_the_root(reg):
    X, y = reg
    Xb = BinMapper().fit(X).transform(X)
    t = HistTree(max_leaf_nodes=64, max_depth=2, min_samples_leaf=1).fit(Xb, -y, np.ones(len(y)))
    assert t.n_leaves == 4                                                  # depth 2 = exactly two levels of splits
    assert HistTree(max_leaf_nodes=64, max_depth=1, min_samples_leaf=1).fit(Xb, -y, np.ones(len(y))).n_leaves == 2


def test_documented_defaults_and_importances_normalised_at_any_scale(reg):
    params = HistGradientBoostingRegressor().get_params()
    assert (params["max_iter"], params["learning_rate"], params["max_leaf_nodes"], params["max_bins"]) == (100, 0.1, 31, 255)
    assert (params["min_samples_leaf"], params["l2_regularization"]) == (20, 0.0)
    X, y = reg
    tiny = HistGradientBoostingRegressor(10, min_samples_leaf=10).fit(X, y * 1e-4)      # total split gain is far below 1
    assert tiny.feature_importances_.sum() == pytest.approx(1.0)


def test_binary_classifier_starts_from_the_logit_of_the_class_prior():
    X = np.random.RandomState(0).randn(200, 3)
    y = np.array([1] * 50 + [0] * 150)                                        # 25% positives
    m = HistGradientBoostingClassifier(1, min_samples_leaf=1000).fit(X, y)    # no split possible: only the baseline remains
    assert m.init_[0] == pytest.approx(np.log(0.25 / 0.75))
    assert np.allclose(m.predict_proba(X)[:, 1], 0.25)
