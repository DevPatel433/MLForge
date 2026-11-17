"""Static plots: decision boundaries, curves, matrices, clusters, trees ..."""
import numpy as np

from ..metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_curve, silhouette_samples
from ._common import PALETTE, class_colors, finish, get_plt, prepare_axes


# ------------------------------------------------------------ classification
def plot_decision_boundary(model, X, y=None, ax=None, resolution=250, proba=False,
                           feature_idx=(0, 1), title=None, save=None, padding=0.6):
    """Colour the plane by the model's prediction (or P(class) with ``proba=True``).
    With more than two features the remaining ones are fixed at their mean."""
    from matplotlib.colors import ListedColormap
    ax = prepare_axes(ax)
    X = np.asarray(X, dtype=float)
    i, j = feature_idx
    x_min, x_max = X[:, i].min() - padding, X[:, i].max() + padding
    y_min, y_max = X[:, j].min() - padding, X[:, j].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.tile(X.mean(axis=0), (xx.size, 1))
    grid[:, i], grid[:, j] = xx.ravel(), yy.ravel()
    if proba and hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid)
        Z = Z[:, 1] if Z.shape[1] == 2 else Z.max(axis=1)
        cs = ax.contourf(xx, yy, Z.reshape(xx.shape), levels=20, cmap="RdBu_r", alpha=0.75)
        ax.figure.colorbar(cs, ax=ax, label="probability")
    else:
        pred = model.predict(grid)
        classes, idx = np.unique(pred, return_inverse=True)
        n_cls = len(np.unique(y)) if y is not None else len(classes)
        cmap = ListedColormap([c for c in class_colors(max(n_cls, len(classes)))])
        ax.contourf(xx, yy, idx.reshape(xx.shape), levels=np.arange(len(classes) + 1) - 0.5,
                    colors=[cmap(k) for k in range(len(classes))], alpha=0.28)
        ax.contour(xx, yy, idx.reshape(xx.shape), levels=np.arange(len(classes)) + 0.5,
                   colors="k", linewidths=0.8, alpha=0.6)
    if y is not None:
        for k, c in enumerate(np.unique(y)):
            m = np.asarray(y) == c
            ax.scatter(X[m, i], X[m, j], s=28, color=PALETTE[k % len(PALETTE)],
                       edgecolor="white", linewidth=0.6, label=str(c))
        ax.legend(frameon=False, title="class")
    return finish(ax, title or f"Decision boundary - {type(model).__name__}", f"feature {i}", f"feature {j}", save)


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=None, ax=None,
                          cmap="Blues", title="Confusion matrix", save=None):
    ax = prepare_axes(ax, (5.2, 4.6))
    labs = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    cm = confusion_matrix(y_true, y_pred, labs, normalize=normalize)
    im = ax.imshow(cm, cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(labs)), [str(l) for l in labs])
    ax.set_yticks(range(len(labs)), [str(l) for l in labs])
    thresh = cm.max() / 2 if cm.max() else 0.5
    for r in range(len(labs)):
        for c in range(len(labs)):
            txt = f"{cm[r, c]:.2f}" if normalize else f"{int(cm[r, c])}"
            ax.text(c, r, txt, ha="center", va="center", color="white" if cm[r, c] > thresh else "#222")
    return finish(ax, title, "predicted", "true", save)


def plot_roc_curve(y_true, scores, ax=None, pos_label=1, title="ROC curve", save=None):
    """``scores`` is an array or a dict ``{model name: scores}`` for comparisons."""
    ax = prepare_axes(ax)
    items = scores if isinstance(scores, dict) else {"model": scores}
    for k, (name, s) in enumerate(items.items()):
        s = np.asarray(s)
        s = s[:, 1] if s.ndim == 2 else s
        fpr, tpr, _ = roc_curve(y_true, s, pos_label)
        ax.plot(fpr, tpr, lw=2, color=PALETTE[k % 10], label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.02)
    return finish(ax, title, "false positive rate", "true positive rate", save, legend=True)


def plot_precision_recall_curve(y_true, scores, ax=None, pos_label=1, title="Precision-recall curve", save=None):
    ax = prepare_axes(ax)
    items = scores if isinstance(scores, dict) else {"model": scores}
    for k, (name, s) in enumerate(items.items()):
        s = np.asarray(s)
        s = s[:, 1] if s.ndim == 2 else s
        p, r, _ = precision_recall_curve(y_true, s, pos_label)
        ax.plot(r, p, lw=2, color=PALETTE[k % 10], label=f"{name} (AP = {average_precision_score(y_true, s, pos_label):.3f})")
    ax.set_ylim(0, 1.02)
    return finish(ax, title, "recall", "precision", save, legend=True)


# --------------------------------------------------------------- training curves
def plot_learning_curve(history, ax=None, log_y=False, title="Cost vs. epoch", ylabel="cost", save=None):
    """``history`` is a list (``model.cost_history_``) or a dict of lists."""
    ax = prepare_axes(ax)
    items = history if isinstance(history, dict) else {"loss": history}
    for k, (name, h) in enumerate(items.items()):
        ax.plot(np.arange(1, len(h) + 1), h, lw=2, color=PALETTE[k % 10], label=name)
    if log_y:
        ax.set_yscale("log")
    return finish(ax, title, "epoch", ylabel, save, legend=len(items) > 1)


def plot_training_curves(history, title="Training curves", save=None):
    """Two-panel loss / metric plot for neural-network ``history_`` dicts."""
    plt = get_plt()
    has_metric = "metric" in history
    fig, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(10 if has_metric else 5.5, 4))
    axes = np.atleast_1d(axes)
    ep = np.arange(1, len(history["loss"]) + 1)
    axes[0].plot(ep, history["loss"], lw=2, label="train", color=PALETTE[0])
    if "val_loss" in history:
        axes[0].plot(ep[:len(history["val_loss"])], history["val_loss"], lw=2, label="validation", color=PALETTE[1])
    finish(axes[0], "Loss", "epoch", "loss", legend=True)
    if has_metric:
        axes[1].plot(ep, history["metric"], lw=2, label="train", color=PALETTE[0])
        if "val_metric" in history:
            axes[1].plot(ep[:len(history["val_metric"])], history["val_metric"], lw=2, label="validation", color=PALETTE[1])
        finish(axes[1], "Accuracy", "epoch", "accuracy", legend=True)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return axes


def plot_model_selection_curve(sizes_or_range, train_scores, val_scores, ax=None, xlabel="training set size",
                               title="Learning curve", save=None, log_x=False):
    """Mean +- std band for learning / validation curves."""
    ax = prepare_axes(ax)
    for scores, name, c in ((train_scores, "train", PALETTE[0]), (val_scores, "validation", PALETTE[1])):
        m, s = scores.mean(1), scores.std(1)
        ax.plot(sizes_or_range, m, "o-", color=c, label=name, lw=2)
        ax.fill_between(sizes_or_range, m - s, m + s, color=c, alpha=0.15)
    if log_x:
        ax.set_xscale("log")
    return finish(ax, title, xlabel, "score", save, legend=True)


def plot_learning_rate_vs_loss(model, X, y, learning_rates=None, ax=None, title="Learning rate vs. final loss", save=None):
    """Train ``model`` once per learning rate and plot the final loss (log-log).
    Diverged runs are drawn at the top, which makes the stable range obvious."""
    ax = prepare_axes(ax)
    lrs = np.logspace(-4, 0.5, 14) if learning_rates is None else np.asarray(learning_rates)
    finals = []
    for lr in lrs:
        m = model.clone().set_params(learning_rate=lr).fit(X, y)
        h = getattr(m, "cost_history_", [np.inf])
        finals.append(h[-1] if len(h) and np.isfinite(h[-1]) else np.nan)
    finals = np.array(finals, dtype=float)
    ok = np.isfinite(finals)
    ax.plot(lrs[ok], finals[ok], "o-", lw=2, color=PALETTE[0])
    if (~ok).any():
        top = np.nanmax(finals[ok]) * 3 if ok.any() else 1
        ax.scatter(lrs[~ok], np.full((~ok).sum(), top), marker="x", color=PALETTE[3], label="diverged")
        ax.legend(frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    return finish(ax, title, "learning rate", "final loss", save)


# ------------------------------------------------------------------- regression
def plot_regression_line(model, X, y, ax=None, title=None, save=None):
    ax = prepare_axes(ax)
    X = np.asarray(X, dtype=float)
    ax.scatter(X[:, 0], y, s=22, alpha=0.7, color=PALETTE[0], edgecolor="white", linewidth=0.5, label="data")
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 300)
    grid = np.tile(X.mean(0), (300, 1))
    grid[:, 0] = xs
    ax.plot(xs, model.predict(grid), color=PALETTE[3], lw=2.5, label="model")
    return finish(ax, title or f"{type(model).__name__} fit", "x", "y", save, legend=True)


def plot_residuals(y_true, y_pred, title="Residual analysis", save=None):
    plt = get_plt()
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, res, s=20, alpha=0.7, color=PALETTE[0], edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color=PALETTE[3], lw=1.5, ls="--")
    finish(axes[0], "Residuals vs. fitted", "fitted value", "residual")
    axes[1].hist(res, bins=25, color=PALETTE[0], edgecolor="white", alpha=0.9)
    finish(axes[1], f"Residual distribution (std = {res.std():.3g})", "residual", "count")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    return axes


def plot_feature_importance(importances, feature_names=None, top_k=15, ax=None,
                            title="Feature importance", save=None):
    ax = prepare_axes(ax, (6.5, 4.8))
    imp = np.asarray(importances, dtype=float).ravel()
    names = feature_names if feature_names is not None else [f"x{i}" for i in range(len(imp))]
    order = np.argsort(imp)[::-1][:top_k][::-1]
    ax.barh([str(names[i]) for i in order], imp[order], color=PALETTE[0])
    return finish(ax, title, "importance", None, save)


def plot_coefficients(coefs, feature_names=None, ax=None, title="Model coefficients", save=None):
    """Signed coefficient bars - handy to compare Ridge / Lasso shrinkage."""
    ax = prepare_axes(ax, (6.5, 4.5))
    coefs = np.asarray(coefs).ravel()
    names = feature_names if feature_names is not None else [f"x{i}" for i in range(len(coefs))]
    ax.barh(range(len(coefs)), coefs, color=[PALETTE[0] if c >= 0 else PALETTE[3] for c in coefs])
    ax.set_yticks(range(len(coefs)), [str(n) for n in names])
    ax.axvline(0, color="k", lw=0.8)
    return finish(ax, title, "coefficient", None, save)


def plot_regularization_path(model_factory, X, y, alphas=None, ax=None, title="Regularisation path", save=None):
    ax = prepare_axes(ax, (6.5, 4.5))
    alphas = np.logspace(-3, 1, 30) if alphas is None else np.asarray(alphas)
    coefs = np.array([model_factory(a).fit(X, y).coef_ for a in alphas])
    for j in range(coefs.shape[1]):
        ax.plot(alphas, coefs[:, j], lw=1.8, label=f"x{j}")
    ax.set_xscale("log")
    return finish(ax, title, "alpha", "coefficient", save, legend=coefs.shape[1] <= 10)


# --------------------------------------------------------------- unsupervised
def plot_clusters(X, labels, centers=None, ax=None, title="Clusters", save=None):
    ax = prepare_axes(ax)
    X = np.asarray(X)
    for k, c in enumerate(np.unique(labels)):
        m = labels == c
        color = "#999999" if c == -1 else PALETTE[k % 10]
        ax.scatter(X[m, 0], X[m, 1], s=26, color=color, edgecolor="white", linewidth=0.5,
                   label="noise" if c == -1 else f"cluster {c}", alpha=0.9)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=190, color="black", edgecolor="white",
                   linewidth=1.2, label="centroids", zorder=5)
    return finish(ax, title, "feature 0", "feature 1", save, legend=True)


def plot_elbow(k_range, inertias, silhouettes=None, ax=None, title="Elbow method", save=None):
    ax = prepare_axes(ax)
    ks = list(k_range)
    ax.plot(ks, inertias, "o-", lw=2, color=PALETTE[0], label="inertia")
    ax.set_ylabel("inertia")
    if silhouettes is not None:
        ax2 = ax.twinx()
        ax2.plot(ks[-len(silhouettes):], silhouettes, "s--", lw=2, color=PALETTE[3], label="silhouette")
        ax2.set_ylabel("silhouette")
    return finish(ax, title, "number of clusters k", None, save)


def plot_silhouette(X, labels, ax=None, title="Silhouette plot", save=None):
    ax = prepare_axes(ax, (6, 5))
    s = silhouette_samples(X, labels)
    y0 = 10
    for k, c in enumerate(np.unique(labels)):
        v = np.sort(s[labels == c])
        ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=PALETTE[k % 10], alpha=0.8)
        ax.text(-0.05, y0 + len(v) / 2, str(c), va="center")
        y0 += len(v) + 10
    ax.axvline(s.mean(), color=PALETTE[3], ls="--", label=f"mean = {s.mean():.3f}")
    ax.set_yticks([])
    return finish(ax, title, "silhouette coefficient", "cluster", save, legend=True)


def plot_pca(X, y=None, pca=None, n_components=2, ax=None, title=None, save=None):
    """Project onto the first 2 or 3 principal components (fits a PCA if none given)."""
    from ..decomposition import PCA
    pca = pca or PCA(n_components).fit(X)
    Z = pca.transform(X)
    k = min(n_components, Z.shape[1])
    if k >= 3:
        ax = prepare_axes(ax, (6.5, 5.5), projection="3d")
    else:
        ax = prepare_axes(ax, (6, 4.8))
    groups = np.unique(y) if y is not None else [None]
    for n, g in enumerate(groups):
        m = np.ones(len(Z), bool) if g is None else np.asarray(y) == g
        args = (Z[m, 0], Z[m, 1]) + ((Z[m, 2],) if k >= 3 else ())
        ax.scatter(*args, s=26, color=PALETTE[n % 10], edgecolor="white", linewidth=0.4,
                   label=None if g is None else str(g), alpha=0.9)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.1%})")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%})")
    if k >= 3:
        ax.set_zlabel(f"PC3 ({ev[2]:.1%})")
    if y is not None:
        ax.legend(frameon=False, title="class")
    return finish(ax, title or f"PCA projection ({k}D, {ev[:k].sum():.1%} variance)", save=save)


def plot_explained_variance(pca_or_ratio, ax=None, title="Explained variance", save=None):
    ax = prepare_axes(ax)
    r = np.asarray(getattr(pca_or_ratio, "explained_variance_ratio_", pca_or_ratio))
    ax.bar(range(1, len(r) + 1), r, color=PALETTE[0], alpha=0.85, label="per component")
    ax.step(range(1, len(r) + 1), np.cumsum(r), where="mid", color=PALETTE[3], lw=2, label="cumulative")
    return finish(ax, title, "principal component", "variance ratio", save, legend=True)


def plot_dendrogram(Z, ax=None, title="Dendrogram", save=None, labels=None):
    """Draw a linkage matrix from :func:`mlforge.cluster.linkage`."""
    ax = prepare_axes(ax, (8, 4.5))
    n = len(Z) + 1
    pos = {}
    order = []

    def leaves(i):
        if i < n:
            return [i]
        a, b = Z[int(i - n), :2]
        return leaves(int(a)) + leaves(int(b))
    order = leaves(2 * n - 2)
    for k, leaf in enumerate(order):
        pos[leaf] = (k, 0.0)
    for step, (a, b, d, _) in enumerate(Z):
        a, b = int(a), int(b)
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.plot([xa, xa, xb, xb], [ya, d, d, yb], color=PALETTE[0], lw=1.4)
        pos[n + step] = ((xa + xb) / 2, d)
    ax.set_xticks(range(n), [str(labels[i]) if labels is not None else str(i) for i in order],
                  fontsize=6, rotation=90)
    return finish(ax, title, None, "distance", save)


def plot_svm(model, X, y, ax=None, resolution=250, title=None, save=None):
    """Decision function contours (margins at +-1), hyperplane and support vectors."""
    ax = prepare_axes(ax)
    X = np.asarray(X, dtype=float)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.7, X[:, 0].max() + 0.7, resolution),
                         np.linspace(X[:, 1].min() - 0.7, X[:, 1].max() + 0.7, resolution))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[Z.min(), 0, Z.max()], colors=[PALETTE[1], PALETTE[0]], alpha=0.18)
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors="k", linestyles=["--", "-", "--"], linewidths=[1, 2, 1])
    for k, c in enumerate(np.unique(y)):
        m = np.asarray(y) == c
        ax.scatter(X[m, 0], X[m, 1], s=28, color=PALETTE[k % 10], edgecolor="white", linewidth=0.5, label=str(c))
    sv = getattr(model, "support_vectors_", None)
    if sv is not None:
        ax.scatter(sv[:, 0], sv[:, 1], s=130, facecolors="none", edgecolors="k", linewidths=1.3, label="support vectors")
    return finish(ax, title or f"{type(model).__name__}: margin & support vectors", "feature 0", "feature 1", save, legend=True)


def plot_benchmark_bars(labels, a, b, a_name="MLForge", b_name="scikit-learn", ax=None,
                        title="Benchmark", ylabel="seconds", log_y=True, save=None):
    ax = prepare_axes(ax, (9, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, a, 0.4, color=PALETTE[0], label=a_name)
    ax.bar(x + 0.2, b, 0.4, color=PALETTE[1], label=b_name)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    if log_y:
        ax.set_yscale("log")
    return finish(ax, title, None, ylabel, save, legend=True)
