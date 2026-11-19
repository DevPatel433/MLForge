"""Smoke tests: every plotting function runs headlessly and returns drawable output."""
import os

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib.pyplot as plt

from mlforge import visualization as viz
from mlforge.cluster import KMeans, elbow_curve, linkage, silhouette_curve
from mlforge.datasets import make_blobs, make_moons
from mlforge.decomposition import PCA
from mlforge.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from mlforge.neural_network import MLPClassifier
from mlforge.optimization import rosenbrock, rosenbrock_grad
from mlforge.svm import SVC
from mlforge.tree import DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def moons():
    return make_moons(120, 0.2, random_state=0)


def test_decision_boundary_variants(moons, tmp_path):
    X, y = moons
    m = LogisticRegression(solver="newton").fit(X, y)
    ax = viz.plot_decision_boundary(m, X, y, save=str(tmp_path / "a.png"))
    assert os.path.getsize(tmp_path / "a.png") > 1000 and ax.get_title()
    viz.plot_decision_boundary(m, X, y, proba=True)
    viz.plot_decision_boundary(DecisionTreeClassifier(max_depth=3).fit(X, y), X)          # no labels
    X5 = np.random.RandomState(0).randn(80, 5)
    y5 = (X5[:, 0] > 0).astype(int)
    viz.plot_decision_boundary(LogisticRegression().fit(X5, y5), X5, y5, feature_idx=(1, 3))   # >2 features
    viz.plot_decision_boundary(RandomForestClassifier(5, random_state=0).fit(X, y), X, y, proba=True)


def test_classification_diagnostics(moons):
    X, y = moons
    m = LogisticRegression(solver="newton").fit(X, y)
    s = m.predict_proba(X)
    viz.plot_confusion_matrix(y, m.predict(X))
    viz.plot_confusion_matrix(y, m.predict(X), normalize="true", labels=[0, 1])
    ax = viz.plot_roc_curve(y, {"a": s, "b": s[:, 1] * 0.9})
    assert len(ax.get_lines()) >= 3
    viz.plot_roc_curve(y, s[:, 1])
    viz.plot_precision_recall_curve(y, {"lr": s})


def test_training_and_selection_curves(moons):
    X, y = moons
    lr = LogisticRegression(n_iters=50).fit(X, y)
    viz.plot_learning_curve(lr.cost_history_, log_y=True)
    viz.plot_learning_curve({"a": [3, 2, 1], "b": [3, 2.5, 2]})
    m = MLPClassifier((8,), n_epochs=5, early_stopping=True, random_state=0).fit(X, y)
    axes = viz.plot_training_curves(m.history_)
    assert len(axes) == 2
    viz.plot_training_curves({"loss": [1, 0.5]})
    sizes = np.array([10, 20, 30])
    viz.plot_model_selection_curve(sizes, np.random.rand(3, 4), np.random.rand(3, 4), log_x=True)
    ax = viz.plot_learning_rate_vs_loss(LinearRegression(solver="gd", n_iters=20), X, y, [1e-3, 1e-1, 1e3, 1e6])
    assert ax.get_xscale() == "log"


def test_regression_plots(reg_data):
    X, y, _ = reg_data
    m = LinearRegression().fit(X, y)
    viz.plot_regression_line(m, X, y)
    axes = viz.plot_residuals(y, m.predict(X))
    assert len(axes) == 2
    viz.plot_feature_importance(np.abs(m.coef_), [f"f{i}" for i in range(5)], top_k=3)
    viz.plot_feature_importance(np.abs(m.coef_))
    viz.plot_coefficients(m.coef_)
    viz.plot_regularization_path(lambda a: Ridge(alpha=a), X, y, alphas=np.logspace(-2, 2, 6))
    viz.plot_regularization_path(lambda a: Lasso(alpha=a), X, y)


def test_unsupervised_plots(tmp_path):
    X, _ = make_blobs(90, 3, random_state=0)
    km = KMeans(3, random_state=0).fit(X)
    viz.plot_clusters(X, km.labels_, km.cluster_centers_)
    viz.plot_clusters(X, np.where(np.arange(90) < 5, -1, km.labels_))              # noise label
    ks = range(2, 5)
    viz.plot_elbow(ks, elbow_curve(X, ks))
    viz.plot_elbow(ks, elbow_curve(X, ks), silhouette_curve(X, ks))
    viz.plot_silhouette(X, km.labels_)
    viz.plot_dendrogram(linkage(X[:25], "average"), labels=list("abcdefghijklmnopqrstuvwxy"))
    viz.plot_dendrogram(linkage(X[:12], "ward"))
    p = PCA(3).fit(X.repeat(2, axis=1)[:, :4])
    viz.plot_pca(X.repeat(2, axis=1), pca=p, n_components=2)
    ax = viz.plot_pca(X.repeat(2, axis=1), np.arange(90) % 3, n_components=3)
    assert hasattr(ax, "set_zlabel")
    viz.plot_pca(X)                                                               # fits its own PCA
    viz.plot_explained_variance(p)
    viz.plot_explained_variance([0.5, 0.3, 0.2])


def test_svm_and_tree_plots(tmp_path):
    X, y = make_blobs(80, [[-2, -2], [2, 2]], random_state=0)
    viz.plot_svm(SVC(kernel="linear", random_state=0).fit(X, y), X, y)
    t = DecisionTreeClassifier(max_depth=3).fit(X, y)
    ax = viz.plot_tree(t, ["a", "b"], ["neg", "pos"], save=str(tmp_path / "t.png"))
    assert os.path.exists(tmp_path / "t.png") and ax.get_title()
    viz.plot_tree(t, max_depth=1)
    viz.plot_tree(DecisionTreeRegressor(max_depth=2).fit(X, X[:, 0]))
    viz.plot_tree(DecisionTreeClassifier().fit(X[:, :1] * 0, y))                  # single-leaf tree


def test_benchmark_bars_and_optimizer_paths(tmp_path):
    viz.plot_benchmark_bars(["a", "b"], [1.0, 2.0], [0.5, 0.1], log_y=False)
    ax, paths = viz.plot_optimizer_paths(rosenbrock, rosenbrock_grad, [-1.0, 1.0],
                                         {"GD": ("gd", 0.001), "Adam": ("adam", 0.05)}, n_iters=50)
    assert set(paths) == {"GD", "Adam"}


def test_animations(tmp_path):
    pytest.importorskip("PIL")
    X, _ = make_blobs(60, 3, random_state=0)
    km = KMeans(3, init="random", n_init=1, random_state=0).fit(X)
    out = viz.animate_kmeans(km, X, path=str(tmp_path / "km.gif"), fps=2)
    assert os.path.getsize(out) > 1000
    out2 = viz.animate_optimizer_paths(rosenbrock, rosenbrock_grad, [-1.0, 1.0], {"Adam": ("adam", 0.1)},
                                       path=str(tmp_path / "opt.gif"), n_iters=40, step=10, fps=5)
    assert os.path.getsize(out2) > 1000


def test_matplotlib_is_lazy_in_top_level_import():
    import subprocess
    import sys
    code = "import sys, mlforge; assert 'matplotlib' not in sys.modules; assert 'sklearn' not in sys.modules; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr
    assert hasattr(__import__("mlforge").visualization, "plot_roc_curve")
