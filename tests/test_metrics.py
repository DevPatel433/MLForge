import numpy as np
import pytest

from mlforge import metrics as M


def test_binary_metrics_by_hand():
    yt = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    yp = np.array([1, 0, 0, 1, 1, 0, 1, 0])       # TP=3 FP=1 FN=1 TN=3
    assert M.accuracy_score(yt, yp) == 0.75 and M.accuracy_score(yt, yp, normalize=False) == 6
    assert M.precision_score(yt, yp) == 0.75 and M.recall_score(yt, yp) == 0.75 and M.f1_score(yt, yp) == 0.75
    assert M.fbeta_score(yt, yp, beta=2) == pytest.approx(0.75)
    cm = M.confusion_matrix(yt, yp)
    assert cm.tolist() == [[3, 1], [1, 3]]
    assert M.confusion_matrix(yt, yp, normalize="true").tolist() == [[0.75, 0.25], [0.25, 0.75]]
    assert M.confusion_matrix(yt, yp, normalize="pred")[0, 0] == 0.75 and M.confusion_matrix(yt, yp, normalize="all").sum() == 1
    assert M.balanced_accuracy_score(yt, yp) == 0.75 and M.matthews_corrcoef(yt, yp) == pytest.approx(0.5)
    assert M.cohen_kappa_score(yt, yp) == pytest.approx(0.5)


def test_degenerate_cases_do_not_crash():
    assert M.precision_score([0, 0], [0, 0]) == 0.0 and M.recall_score([0, 0], [0, 0]) == 0.0 and M.f1_score([0, 0], [0, 0]) == 0.0
    assert M.matthews_corrcoef([1, 1], [1, 1]) == 0.0
    with pytest.raises(ValueError):
        M.accuracy_score([1, 2], [1])
    with pytest.raises(ValueError, match="binary"):
        M.precision_score([0, 1, 2], [0, 1, 2])
    with pytest.raises(ValueError):
        M.roc_auc_score([1, 1, 1], [0.2, 0.5, 0.9])


@pytest.mark.parametrize("avg,expected", [("macro", 0.6032), ("micro", 0.6), ("weighted", 0.6)])
def test_multiclass_averaging(avg, expected):
    yt = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2])
    yp = np.array([0, 1, 1, 1, 2, 0, 2, 0, 2, 1])
    # per-class F1 by hand: 2/3, 4/7, 4/7  ->  macro 0.6032, micro = accuracy = 0.6, weighted 0.6
    from sklearn import metrics as S
    assert M.f1_score(yt, yp, average=avg) == pytest.approx(S.f1_score(yt, yp, average=avg))
    assert M.f1_score(yt, yp, average=avg) == pytest.approx(expected, abs=1e-3)
    p, r, f, s = M.precision_recall_fscore_support(yt, yp)
    assert s.tolist() == [3, 3, 4] and len(p) == 3


def test_roc_auc_and_curves():
    yt = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    assert M.roc_auc_score(yt, s) == 0.75
    fpr, tpr, th = M.roc_curve(yt, s)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1 and np.all(np.diff(th) <= 0)
    assert M.auc(fpr, tpr) == pytest.approx(0.75)
    assert M.roc_auc_score(yt, np.zeros(4)) == 0.5                       # ties => chance
    assert M.roc_auc_score(yt, -s) == pytest.approx(0.25)
    assert M.average_precision_score(yt, s) == pytest.approx(0.8333, abs=1e-3)
    p, r, _ = M.precision_recall_curve(yt, s)
    assert p[0] == 1 and r[0] == 0


def test_multiclass_auc_and_log_loss():
    yt = np.array([0, 1, 2, 0, 1, 2])
    P = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .2, .7], [.6, .3, .1], [.1, .8, .1], [.2, .2, .6]])
    assert M.roc_auc_score(yt, P) == 1.0
    ll = M.log_loss(yt, P)
    assert ll == pytest.approx(-np.mean(np.log(P[np.arange(6), yt])))
    assert M.log_loss([0, 1], [0.1, 0.9]) == pytest.approx(-np.log(0.9))
    assert M.log_loss([0, 1], np.array([[0.9, 0.1], [0.1, 0.9]])) == pytest.approx(-np.log(0.9))
    assert np.isfinite(M.log_loss([0, 1], [0.0, 1.0]))                    # clipping
    assert M.brier_score_loss([0, 1], [0.2, 0.8]) == pytest.approx(0.04)


def test_classification_report_text():
    rep = M.classification_report([0, 1, 1, 0, 2], [0, 1, 0, 0, 2])
    for token in ("precision", "recall", "f1-score", "accuracy", "macro avg", "weighted avg"):
        assert token in rep


def test_regression_metrics_by_hand():
    yt, yp = np.array([3.0, -0.5, 2.0, 7.0]), np.array([2.5, 0.0, 2.0, 8.0])
    assert M.mean_absolute_error(yt, yp) == 0.5 and M.mean_squared_error(yt, yp) == 0.375
    assert M.root_mean_squared_error(yt, yp) == pytest.approx(np.sqrt(0.375))
    assert M.r2_score(yt, yp) == pytest.approx(0.9486, abs=1e-4)
    assert M.median_absolute_error(yt, yp) == 0.5 and M.max_error(yt, yp) == 1.0
    assert M.explained_variance_score(yt, yp) == pytest.approx(0.9572, abs=1e-4)
    assert M.mean_absolute_percentage_error([1.0, 2.0], [1.1, 1.8]) == pytest.approx(0.1)
    assert M.r2_score([1, 1], [1, 1]) == 1.0 and M.r2_score([1, 1], [1, 2]) == 0.0
    assert M.r2_score(yt, np.full(4, yt.mean())) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        M.mean_squared_error([1, 2], [1])


def test_cluster_metrics(blobs3):
    X, y = blobs3
    assert M.silhouette_score(X, y) > 0.6 and M.silhouette_samples(X, y).shape == (len(X),)
    assert M.davies_bouldin_score(X, y) < 0.6 and M.calinski_harabasz_score(X, y) > 100
    rng = np.random.RandomState(0)
    shuffled = rng.permutation(y)
    assert M.silhouette_score(X, shuffled) < 0.1
    assert M.adjusted_rand_score(y, (y + 1) % 3) == pytest.approx(1.0)      # label permutation invariant
    assert abs(M.adjusted_rand_score(y, shuffled)) < 0.05
    assert M.normalized_mutual_info_score(y, (y + 2) % 3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        M.silhouette_score(X, np.zeros(len(X)))


# ---- tests added after mutation testing exposed gaps -------------------------------------------
def test_explained_variance_score_on_constant_target_is_zero_not_nan():
    # var(y_true) == 0 would divide by zero; the metric must return 0.0 rather than nan / inf
    assert M.explained_variance_score([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    assert M.explained_variance_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0


def test_fbeta_precision_and_recall_differ_when_they_should():
    # TP=1, FP=1, FN=2  ->  precision = 1/2, recall = 1/3  (so recall != F1 != precision)
    yt = np.array([1, 1, 1, 0, 0, 0])
    yp = np.array([1, 0, 0, 0, 0, 1])
    assert M.precision_score(yt, yp) == pytest.approx(1 / 2)
    assert M.recall_score(yt, yp) == pytest.approx(1 / 3)
    assert M.f1_score(yt, yp) == pytest.approx(0.4)
    assert M.fbeta_score(yt, yp, beta=2) == pytest.approx(5 * (1 / 2) * (1 / 3) / (4 * (1 / 2) + 1 / 3))     # 0.35714
    assert M.fbeta_score(yt, yp, beta=0.5) == pytest.approx(1.25 * (1 / 2) * (1 / 3) / (0.25 * (1 / 2) + 1 / 3))  # 0.4545
    # micro averaging pools the counts: for single-label problems precision = recall = F1 = accuracy
    for fn in (M.precision_score, M.recall_score, M.f1_score):
        assert fn(yt, yp, average="micro") == pytest.approx(M.accuracy_score(yt, yp))


def test_roc_curve_threshold_values():
    s = np.array([0.1, 0.4, 0.35, 0.8])
    _, _, th = M.roc_curve([0, 0, 1, 1], s)
    assert th[0] == pytest.approx(s.max() + 1)                   # sentinel threshold above every score => (0, 0) point
    assert th[1:].tolist() == [0.8, 0.4, 0.35, 0.1]              # then each distinct score, decreasing


def test_silhouette_singleton_cluster_scores_zero():
    # a cluster with one member has no intra-cluster distance; by convention its silhouette is 0
    X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]])
    labels = np.array([0, 0, 0, 1])
    s = M.silhouette_samples(X, labels)
    assert s[3] == 0.0 and np.all(s[:3] > 0.9)
    from sklearn import metrics as S
    assert np.allclose(s, S.silhouette_samples(X, labels))


def test_calinski_harabasz_degenerate_perfect_clusters_returns_one():
    # every cluster collapses to a single location: within-cluster dispersion is exactly 0
    X = np.array([[0.0, 0.0], [0.0, 0.0], [3.0, 3.0], [3.0, 3.0]])
    assert M.calinski_harabasz_score(X, np.array([0, 0, 1, 1])) == 1.0


def test_calinski_harabasz_is_scale_invariant_including_tiny_dispersion():
    # with tiny within-cluster dispersion (0 < intra < 1) the metric must still be computed, not replaced by 1.0
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(30, 2), rng.randn(30, 2) + 6])
    labels = np.array([0] * 30 + [1] * 30)
    base = M.calinski_harabasz_score(X, labels)
    assert M.calinski_harabasz_score(X * 0.01, labels) == pytest.approx(base) and base > 10


def test_fbeta_with_macro_and_weighted_averaging_matches_scikit_learn():
    from sklearn import metrics as S
    yt = np.array([0, 0, 1, 1, 2, 2, 2, 0, 1, 2])
    yp = np.array([0, 1, 1, 1, 2, 0, 2, 0, 2, 1])
    for beta in (0.5, 2.0):
        for avg in ("macro", "weighted", "micro"):
            assert M.fbeta_score(yt, yp, beta, average=avg) == pytest.approx(S.fbeta_score(yt, yp, beta=beta, average=avg))
