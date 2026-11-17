"""Generate the gallery images used in the README / docs (docs/images/*.png).

    python examples/make_gallery.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlforge import visualization as viz
from mlforge.cluster import AgglomerativeClustering, KMeans, elbow_curve, linkage, silhouette_curve
from mlforge.datasets import make_blobs, make_circles, make_moons, make_regression
from mlforge.decomposition import PCA
from mlforge.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from mlforge.model_selection import learning_curve, train_test_split
from mlforge.neighbors import KNeighborsClassifier
from mlforge.neural_network import MLPClassifier
from mlforge.optimization import rosenbrock, rosenbrock_grad
from mlforge.preprocessing import StandardScaler
from mlforge.svm import SVC
from mlforge.tree import DecisionTreeClassifier, RandomForestClassifier

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
path = lambda name: os.path.join(OUT, name)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "font.size": 9})


def decision_boundaries():
    X, y = make_moons(300, 0.22, random_state=1)
    models = {"Logistic Regression": LogisticRegression(solver="newton"),
              "KNN (k=15)": KNeighborsClassifier(15),
              "Decision Tree (depth 5)": DecisionTreeClassifier(max_depth=5),
              "Random Forest": RandomForestClassifier(50, random_state=0),
              "SVM (RBF, SMO)": SVC(C=2.0, random_state=0),
              "MLP (2x16)": MLPClassifier((16, 16), n_epochs=150, random_state=0)}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, m) in zip(axes.ravel(), models.items()):
        m.fit(X, y)
        viz.plot_decision_boundary(m, X, y, ax=ax, title=f"{name}\nacc = {m.score(X, y):.3f}")
    fig.tight_layout()
    fig.savefig(path("decision_boundaries.png"), dpi=130)


def knn_k():
    X, y = make_moons(200, 0.3, random_state=2)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, k in zip(axes, [1, 5, 15, 50]):
        viz.plot_decision_boundary(KNeighborsClassifier(k).fit(X, y), X, y, ax=ax, title=f"k = {k}")
        ax.get_legend().remove()
    fig.tight_layout()
    fig.savefig(path("knn_k_values.png"), dpi=130)


def optimizer_race():
    opts = {"GD": ("gd", 0.0015), "Momentum": ("momentum", 0.0004), "Nesterov": ("nesterov", 0.0004),
            "AdaGrad": ("adagrad", 0.6), "RMSProp": ("rmsprop", 0.02), "Adam": ("adam", 0.1)}
    viz.plot_optimizer_paths(rosenbrock, rosenbrock_grad, [-1.5, 2.0], opts, xlim=(-2, 2), ylim=(-0.8, 3.2),
                             n_iters=2500, title="Optimizers on the Rosenbrock valley",
                             save=path("optimizer_race.png"))
    viz.animate_optimizer_paths(rosenbrock, rosenbrock_grad, [-1.5, 2.0], opts, path=path("optimizer_race.gif"),
                                xlim=(-2, 2), ylim=(-0.8, 3.2), n_iters=1500, step=25, fps=12)


def kmeans_gif():
    X, _ = make_blobs(300, 4, cluster_std=1.1, random_state=7)
    km = KMeans(4, init="random", n_init=1, random_state=3).fit(X)
    viz.animate_kmeans(km, X, path=path("kmeans.gif"), fps=2)
    ks = list(range(1, 9))
    viz.plot_elbow(ks, elbow_curve(X, ks), silhouette_curve(X, range(2, 9)), save=path("elbow.png"))
    viz.plot_clusters(X, km.labels_, km.cluster_centers_, title="K-Means (k = 4)", save=path("kmeans_clusters.png"))


def pca_plots():
    from sklearn.datasets import load_wine
    X, y = load_wine(return_X_y=True)
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(3).fit(Xs)
    viz.plot_pca(Xs, y, pca=pca, n_components=2, save=path("pca_2d.png"))
    ax = viz.plot_pca(Xs, y, pca=pca, n_components=3)
    ax.figure.savefig(path("pca_3d.png"), dpi=130)
    viz.plot_explained_variance(PCA(None).fit(Xs), save=path("pca_variance.png"))


def tree_plot():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    viz.plot_tree(t, ["sepal len", "sepal wid", "petal len", "petal wid"], ["setosa", "versicolor", "virginica"],
                  save=path("decision_tree.png"), fontsize=8)


def classification_metrics():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    models = {"Logistic": LogisticRegression(solver="newton", C=1.0),
              "Random Forest": RandomForestClassifier(60, random_state=0),
              "SVM (RBF)": SVC(random_state=0)}
    scores = {}
    for n, m in models.items():
        m.fit(Xtr, ytr)
        scores[n] = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    viz.plot_roc_curve(yte, scores, ax=axes[0])
    viz.plot_precision_recall_curve(yte, scores, ax=axes[1])
    viz.plot_confusion_matrix(yte, models["Logistic"].predict(Xte), labels=[0, 1], ax=axes[2],
                              title="Confusion matrix (Logistic)")
    fig.tight_layout()
    fig.savefig(path("classification_metrics.png"), dpi=130)


def regression_plots():
    X, y = make_regression(120, 1, noise=7.0, bias=4.0, random_state=0)
    m = LinearRegression(solver="gd", n_iters=300, learning_rate=0.1).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    viz.plot_regression_line(m, X, y, ax=axes[0], title="Linear regression (gradient descent)")
    viz.plot_learning_curve(m.cost_history_, ax=axes[1], log_y=True, title="Cost convergence", ylabel="MSE")
    fig.tight_layout()
    fig.savefig(path("regression.png"), dpi=130)
    viz.plot_residuals(y, m.predict(X), save=path("residuals.png"))
    Xr = np.random.RandomState(0).randn(150, 8)
    w = np.array([4, -3, 0, 0, 2, 0, 0, -1.0])
    yr = Xr @ w + np.random.RandomState(1).randn(150) * 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    viz.plot_regularization_path(lambda a: Ridge(alpha=a), Xr, yr, alphas=np.logspace(-2, 3, 30), ax=axes[0], title="Ridge path")
    viz.plot_regularization_path(lambda a: Lasso(alpha=a), Xr, yr, alphas=np.logspace(-3, 0.5, 30),
                                 ax=axes[1], title="Lasso path (sparsity)")
    fig.tight_layout()
    fig.savefig(path("regularization_paths.png"), dpi=130)
    lr_model = LinearRegression(solver="gd", n_iters=60)
    viz.plot_learning_rate_vs_loss(lr_model, StandardScaler().fit_transform(Xr), yr,
                                   np.logspace(-3, 0.7, 14), save=path("lr_vs_loss.png"))


def svm_plot():
    X, y = make_blobs(120, [[-2, -2], [2, 2]], cluster_std=1.1, random_state=4)
    m = SVC(kernel="linear", C=1.0, random_state=0).fit(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    viz.plot_svm(m, X, y, ax=axes[0], title="Linear SVM (SMO)")
    Xc, yc = make_circles(200, 0.07, random_state=0)
    viz.plot_svm(SVC(C=5, random_state=0).fit(Xc, yc), Xc, yc, ax=axes[1], title="RBF-kernel SVM")
    fig.tight_layout()
    fig.savefig(path("svm.png"), dpi=130)


def clustering_gallery():
    Xm, _ = make_moons(250, 0.07, random_state=0)
    from mlforge.cluster import DBSCAN, GaussianMixture
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    viz.plot_clusters(Xm, KMeans(2, random_state=0).fit(Xm).labels_, ax=axes[0], title="K-Means")
    viz.plot_clusters(Xm, AgglomerativeClustering(2, "single").fit(Xm).labels_, ax=axes[1], title="Single linkage")
    viz.plot_clusters(Xm, DBSCAN(0.18, 5).fit(Xm).labels_, ax=axes[2], title="DBSCAN")
    Xb, _ = make_blobs(200, 3, cluster_std=1.3, random_state=5)
    viz.plot_clusters(Xb, GaussianMixture(3, random_state=0).fit(Xb).predict(Xb), ax=axes[3], title="Gaussian mixture (EM)")
    for a in axes:
        a.get_legend().remove()
    fig.tight_layout()
    fig.savefig(path("clustering_gallery.png"), dpi=130)
    Xd, _ = make_blobs(30, 3, cluster_std=0.9, random_state=1)
    viz.plot_dendrogram(linkage(Xd, "ward"), save=path("dendrogram.png"), title="Ward dendrogram")


def neural_net_plot():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    m = MLPClassifier((64,), n_epochs=60, learning_rate=0.003, l2=1e-4, dropout=0.1, early_stopping=True,
                      patience=10, random_state=0).fit(sc.transform(Xtr), ytr)
    print("digits accuracy", m.score(sc.transform(Xte), yte))
    viz.plot_training_curves(m.history_, title="MLP on handwritten digits", save=path("training_curves.png"))
    viz.plot_confusion_matrix(yte, m.predict(sc.transform(Xte)), title="Digits confusion matrix", save=path("digits_confusion.png"))


def importance_plot():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer()
    rf = RandomForestClassifier(60, random_state=0, oob_score=True).fit(d.data, d.target)
    viz.plot_feature_importance(rf.feature_importances_, d.feature_names, top_k=10,
                                title=f"Random-forest importance (OOB acc = {rf.oob_score_:.3f})",
                                save=path("feature_importance.png"))


def learning_curve_plot():
    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    sizes, tr, va = learning_curve(LogisticRegression(solver="newton", C=1.0), X, y, cv=5)
    viz.plot_model_selection_curve(sizes, tr, va, save=path("learning_curve.png"),
                                   title="Learning curve - Logistic Regression")


if __name__ == "__main__":
    steps = [decision_boundaries, knn_k, optimizer_race, kmeans_gif, pca_plots, tree_plot, classification_metrics,
             regression_plots, svm_plot, clustering_gallery, neural_net_plot, importance_plot, learning_curve_plot]
    only = sys.argv[1:]
    for fn in steps:
        if only and fn.__name__ not in only:
            continue
        print("->", fn.__name__, flush=True)
        fn()
        plt.close("all")
    print("gallery written to", os.path.abspath(OUT))
