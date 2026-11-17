"""Benchmark every MLForge algorithm against its scikit-learn equivalent.

For each algorithm the harness records

* **fit time** and **predict time** (median of several repeats),
* **peak memory** during ``fit`` (``tracemalloc``),
* **quality** on held-out data (accuracy, R^2 or adjusted Rand index),
* **numerical difference** between the two implementations' outputs
  (max absolute difference of probabilities / predictions, relative inertia
  difference, ... - see each case).

scikit-learn is an *optional* dependency used **only here**: the library
itself never imports it. Run with ``python -m mlforge benchmark``.
"""
import csv
import gc
import json
import os
import time
import tracemalloc
import warnings

import numpy as np


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Benchmarking needs scikit-learn: pip install scikit-learn") from exc


# ------------------------------------------------------------------- datasets
def _datasets(quick):
    from sklearn.datasets import load_breast_cancer, load_digits, load_wine

    from .datasets import make_blobs, make_moons
    from .preprocessing import StandardScaler, train_test_split
    rng = np.random.RandomState(0)
    n = 700 if quick else 2000
    Xr = rng.randn(n, 20)
    w = rng.randn(20)
    yr = Xr @ w + 2 * np.sin(2 * Xr[:, 0]) + rng.randn(n) * 0.5
    Xb = rng.randn(n, 20)
    yb = (Xb @ rng.randn(20) + rng.logistic(size=n) > 0).astype(int)
    out = {}
    for name, (X, y) in {"regression": (Xr, yr), "binary": (Xb, yb),
                         "cancer": load_breast_cancer(return_X_y=True),
                         "wine": load_wine(return_X_y=True), "digits": load_digits(return_X_y=True)}.items():
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0,
                                              stratify=None if name == "regression" else y)
        if name != "digits":
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        out[name] = (Xtr, Xte, ytr, yte)
    Xc, yc = make_blobs(600 if quick else 1500, 5, cluster_std=1.1, random_state=3)
    out["clusters"] = (Xc, Xc, yc, yc)
    Xm, ym = make_moons(600 if quick else 1500, 0.06, random_state=3)
    out["moons"] = (Xm, Xm, ym, ym)
    Xp = rng.randn(n, 30) @ rng.randn(30, 30)
    out["pca"] = (Xp, Xp, None, None)
    return out


# ---------------------------------------------------------------------- cases
def _cases(quick):
    """(name, family, dataset, ours, reference, kind, diff_kind) tuples built lazily."""
    from sklearn import cluster as skc
    from sklearn import decomposition as skd
    from sklearn import ensemble as ske
    from sklearn import linear_model as sklm
    from sklearn import mixture as skmix
    from sklearn import naive_bayes as sknb
    from sklearn import neighbors as skn
    from sklearn import neural_network as sknn
    from sklearn import svm as sksvm
    from sklearn import tree as skt

    from . import bayes, cluster, decomposition, neighbors, neural_network, svm, tree
    from . import linear_model as lm
    n_est = 25 if quick else 60
    it_gd = 4000 if quick else 8000          # first-order solvers need many steps to reach the exact optimum
    reg, clf, unsup = "regression", "classification", "unsupervised"
    return [
        # ---- regression -------------------------------------------------------
        ("LinearRegression (closed form)", reg, "regression", lambda: lm.LinearRegression(), lambda: sklm.LinearRegression(), "predict_reg"),
        ("LinearRegression (gradient descent)", reg, "regression", lambda: lm.LinearRegression(solver="gd", learning_rate=0.1, n_iters=it_gd // 2, optimizer="adam"), lambda: sklm.LinearRegression(), "predict_reg"),
        ("Ridge", reg, "regression", lambda: lm.Ridge(alpha=2.0), lambda: sklm.Ridge(alpha=2.0), "predict_reg"),
        ("Lasso (coordinate descent)", reg, "regression", lambda: lm.Lasso(alpha=0.05, tol=1e-9), lambda: sklm.Lasso(alpha=0.05, tol=1e-9), "predict_reg"),
        ("ElasticNet", reg, "regression", lambda: lm.ElasticNet(alpha=0.05, l1_ratio=0.5, tol=1e-9), lambda: sklm.ElasticNet(alpha=0.05, l1_ratio=0.5, tol=1e-9), "predict_reg"),
        ("KNeighborsRegressor", reg, "regression", lambda: neighbors.KNeighborsRegressor(5), lambda: skn.KNeighborsRegressor(5), "predict_reg"),
        ("DecisionTreeRegressor", reg, "regression", lambda: tree.DecisionTreeRegressor(max_depth=8), lambda: skt.DecisionTreeRegressor(max_depth=8, random_state=0), "predict_reg"),
        ("RandomForestRegressor", reg, "regression", lambda: tree.RandomForestRegressor(n_est, max_depth=10, random_state=0), lambda: ske.RandomForestRegressor(n_est, max_depth=10, max_features=1.0, random_state=0), "predict_reg"),
        ("GradientBoostingRegressor", reg, "regression", lambda: tree.GradientBoostingRegressor(n_est, random_state=0), lambda: ske.GradientBoostingRegressor(n_estimators=n_est, random_state=0), "predict_reg"),
        ("HistGradientBoostingRegressor", reg, "regression", lambda: tree.HistGradientBoostingRegressor(n_est), lambda: ske.HistGradientBoostingRegressor(max_iter=n_est, early_stopping=False, random_state=0), "predict_reg"),
        ("MLPRegressor", reg, "regression", lambda: neural_network.MLPRegressor((32,), n_epochs=60, random_state=0), lambda: sknn.MLPRegressor(hidden_layer_sizes=(32,), max_iter=60, random_state=0), "predict_reg"),
        # ---- classification ---------------------------------------------------
        ("LogisticRegression (Newton)", clf, "cancer", lambda: lm.LogisticRegression(solver="newton", C=1.0), lambda: sklm.LogisticRegression(C=1.0, tol=1e-12, max_iter=10000), "proba"),
        ("LogisticRegression (gradient descent)", clf, "cancer", lambda: lm.LogisticRegression(solver="gd", C=1.0, learning_rate=0.5, n_iters=it_gd), lambda: sklm.LogisticRegression(C=1.0, tol=1e-12, max_iter=10000), "proba"),
        ("SoftmaxRegression", clf, "wine", lambda: lm.SoftmaxRegression(C=1.0, learning_rate=0.05, n_iters=it_gd, optimizer="adam"), lambda: sklm.LogisticRegression(C=1.0, tol=1e-12, max_iter=10000), "proba"),
        ("Perceptron", clf, "binary", lambda: lm.Perceptron(random_state=0), lambda: sklm.Perceptron(random_state=0), "predict_cls"),
        ("KNeighborsClassifier", clf, "digits", lambda: neighbors.KNeighborsClassifier(5), lambda: skn.KNeighborsClassifier(5), "proba"),
        ("GaussianNB", clf, "cancer", lambda: bayes.GaussianNB(), lambda: sknb.GaussianNB(), "proba"),
        ("DecisionTreeClassifier", clf, "cancer", lambda: tree.DecisionTreeClassifier(max_depth=6), lambda: skt.DecisionTreeClassifier(max_depth=6, random_state=0), "predict_cls"),
        ("RandomForestClassifier", clf, "digits", lambda: tree.RandomForestClassifier(n_est, random_state=0), lambda: ske.RandomForestClassifier(n_est, random_state=0), "predict_cls"),
        ("GradientBoostingClassifier", clf, "cancer", lambda: tree.GradientBoostingClassifier(n_est, random_state=0), lambda: ske.GradientBoostingClassifier(n_estimators=n_est, random_state=0), "proba"),
        ("HistGradientBoostingClassifier", clf, "digits", lambda: tree.HistGradientBoostingClassifier(n_est), lambda: ske.HistGradientBoostingClassifier(max_iter=n_est, early_stopping=False, random_state=0), "predict_cls"),
        ("SVC (RBF kernel, SMO)", clf, "cancer", lambda: svm.SVC(C=1.0, random_state=0), lambda: sksvm.SVC(C=1.0), "predict_cls"),
        ("LinearSVC", clf, "binary", lambda: svm.LinearSVC(C=1.0, n_iters=500, loss="squared_hinge"), lambda: sksvm.LinearSVC(C=1.0, max_iter=20000), "predict_cls"),
        ("MLPClassifier", clf, "digits", lambda: neural_network.MLPClassifier((64,), n_epochs=30, learning_rate=0.003, random_state=0), lambda: sknn.MLPClassifier(hidden_layer_sizes=(64,), max_iter=30, random_state=0), "predict_cls"),
        # ---- unsupervised -----------------------------------------------------
        ("KMeans (k-means++)", unsup, "clusters", lambda: cluster.KMeans(5, n_init=5, random_state=0), lambda: skc.KMeans(5, n_init=5, random_state=0), "cluster"),
        ("AgglomerativeClustering (ward)", unsup, "clusters", lambda: cluster.AgglomerativeClustering(5, "ward"), lambda: skc.AgglomerativeClustering(5, linkage="ward"), "cluster"),
        ("DBSCAN", unsup, "moons", lambda: cluster.DBSCAN(0.12, 5), lambda: skc.DBSCAN(eps=0.12, min_samples=5), "cluster"),
        ("GaussianMixture (EM)", unsup, "clusters", lambda: cluster.GaussianMixture(5, random_state=0), lambda: skmix.GaussianMixture(5, random_state=0), "cluster"),
        ("PCA (SVD)", unsup, "pca", lambda: decomposition.PCA(5), lambda: skd.PCA(5), "pca"),
    ]


# --------------------------------------------------------------------- engine
def _timed(fn, repeats):
    times, result = [], None
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def _peak_mb(fn):
    gc.collect()
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def _quality(kind, family, model, X, y, labels=None):
    from . import metrics as M
    if kind == "cluster":
        lab = labels if labels is not None else model.labels_
        return float(M.adjusted_rand_score(y, lab))
    if kind == "pca":
        return float(np.sum(model.explained_variance_ratio_))
    pred = model.predict(X)
    return float(M.r2_score(y, pred) if family == "regression" else M.accuracy_score(y, pred))


DIFF_LABELS = {
    "predict_reg": "max abs. \u0394 prediction",
    "predict_cls": "label disagreement rate",
    "proba": "max abs. \u0394 probability",
    "cluster": "1 - adjusted Rand index",
    "pca": "max abs. \u0394 variance ratio",
}


def _difference(kind, ours, ref, X, a_labels=None, b_labels=None):
    from . import metrics as M
    if kind == "predict_reg":
        return float(np.max(np.abs(np.asarray(ours.predict(X), float) - np.asarray(ref.predict(X), float))))
    if kind == "predict_cls":
        return float(np.mean(np.asarray(ours.predict(X)) != np.asarray(ref.predict(X))))
    if kind == "proba":
        return float(np.max(np.abs(ours.predict_proba(X) - ref.predict_proba(X))))
    if kind == "cluster":
        return float(max(0.0, 1.0 - M.adjusted_rand_score(a_labels, b_labels)))
    if kind == "pca":
        return float(np.max(np.abs(ours.explained_variance_ratio_ - ref.explained_variance_ratio_)))
    return float("nan")


def run_benchmarks(quick=False, repeats=3, only=None, verbose=True):
    """Run the whole suite and return a list of result dicts."""
    _require_sklearn()
    warnings.filterwarnings("ignore")
    data = _datasets(quick)
    rows = []
    for name, family, ds, make_ours, make_ref, kind in _cases(quick):
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        Xtr, Xte, ytr, yte = data[ds]
        supervised = ytr is not None and family != "unsupervised"
        row = {"algorithm": name, "family": family, "dataset": ds, "n_train": len(Xtr), "n_features": Xtr.shape[1]}
        try:
            fit = (lambda m, X=Xtr, y=ytr: m.fit(X, y)) if supervised else (
                (lambda m, X=Xtr: m.fit(X)) if family == "unsupervised" else None)
            reps = 1 if ("gradient descent" in name or "MLP" in name or "Forest" in name or "Boosting" in name) else repeats
            for tag, make in (("mlforge", make_ours), ("sklearn", make_ref)):
                t_fit, model = _timed(lambda: fit(make()), reps)
                mem = _peak_mb(lambda: fit(make()))
                if kind == "cluster":
                    labels = getattr(model, "labels_", None)
                    if labels is None:
                        labels = model.predict(Xtr)
                    if tag == "sklearn":
                        labels_ref = labels
                    else:
                        labels_ours = labels
                    t_pred = 0.0
                elif kind == "pca":
                    t_pred, _ = _timed(lambda: model.transform(Xte), repeats)
                else:
                    t_pred, _ = _timed(lambda: model.predict(Xte), repeats)
                q = _quality(kind, family, model, Xte, yte, labels if kind == "cluster" else None)
                row.update({f"{tag}_fit_s": t_fit, f"{tag}_predict_s": t_pred, f"{tag}_memory_mb": mem, f"{tag}_quality": q})
                if tag == "mlforge":
                    ours_model = model
                else:
                    ref_model = model
            row["max_abs_diff"] = _difference(kind, ours_model, ref_model, Xte,
                                              labels_ours if kind == "cluster" else None,
                                              labels_ref if kind == "cluster" else None)
            row["slowdown_fit"] = row["mlforge_fit_s"] / max(row["sklearn_fit_s"], 1e-9)
            row["difference_metric"] = DIFF_LABELS[kind]
            row["quality_metric"] = "R2" if family == "regression" else ("ARI" if kind == "cluster" else (
                "explained variance" if kind == "pca" else "accuracy"))
            row["status"] = "ok"
        except Exception as exc:  # keep the suite going; report the failure
            row["status"] = f"error: {type(exc).__name__}: {exc}"
        rows.append(row)
        if verbose:
            if row["status"] == "ok":
                print(f"  {name:<40} fit {row['mlforge_fit_s']:.3f}s vs {row['sklearn_fit_s']:.3f}s "
                      f"| {row['quality_metric']} {row['mlforge_quality']:.4f} vs {row['sklearn_quality']:.4f} "
                      f"| {row['difference_metric']} {row['max_abs_diff']:.2e}", flush=True)
            else:
                print(f"  {name:<40} {row['status']}", flush=True)
    return rows


# --------------------------------------------------------------------- output
def _fmt_time(s):
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


def to_markdown(rows):
    head = ("| Algorithm | Data | Fit (MLForge) | Fit (sklearn) | Slowdown | Predict (MLForge) | Predict (sklearn) "
            "| Peak mem (MLForge / sklearn) | Quality (MLForge / sklearn) | Difference from scikit-learn |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    body = ""
    for r in rows:
        if r["status"] != "ok":
            body += f"| {r['algorithm']} | {r['dataset']} | {r['status']} |||||||\n"
            continue
        body += (f"| {r['algorithm']} | {r['dataset']} | {_fmt_time(r['mlforge_fit_s'])} | {_fmt_time(r['sklearn_fit_s'])} "
                 f"| {r['slowdown_fit']:.1f}x | {_fmt_time(r['mlforge_predict_s'])} | {_fmt_time(r['sklearn_predict_s'])} "
                 f"| {r['mlforge_memory_mb']:.1f} / {r['sklearn_memory_mb']:.1f} MB "
                 f"| {r['quality_metric']} {r['mlforge_quality']:.4f} / {r['sklearn_quality']:.4f} "
                 f"| {r['max_abs_diff']:.1e} ({r['difference_metric']}) |\n")
    return head + body


def save_results(rows, out_dir, plots=True):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "benchmark_results.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "benchmark_results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "benchmark_results.md"), "w") as fh:
        fh.write("# MLForge vs scikit-learn benchmark\n\n")
        fh.write(to_markdown(rows))
    if plots:
        try:
            _plots(rows, out_dir)
        except ImportError:  # pragma: no cover - matplotlib optional
            pass
    return out_dir


def _plots(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .visualization import plot_benchmark_bars
    ok = [r for r in rows if r["status"] == "ok"]
    names = [r["algorithm"] for r in ok]
    ax = plot_benchmark_bars(names, [r["mlforge_fit_s"] for r in ok], [r["sklearn_fit_s"] for r in ok],
                             title="Training time (lower is better, log scale)", ylabel="seconds")
    ax.figure.set_size_inches(13, 5.5)
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(out_dir, "benchmark_fit_time.png"), dpi=130)
    plt.close(ax.figure)
    ax = plot_benchmark_bars(names, [max(r["max_abs_diff"], 1e-17) for r in ok], [1e-17] * len(ok),
                             "max |difference|", "", title="Numerical difference from scikit-learn (log scale)",
                             ylabel="max abs. difference of outputs")
    ax.figure.set_size_inches(13, 5.5)
    ax.get_legend().remove()
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(out_dir, "benchmark_difference.png"), dpi=130)
    plt.close(ax.figure)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(ok))
    ax.bar(x - 0.2, [r["mlforge_quality"] for r in ok], 0.4, label="MLForge", color="#4C72B0")
    ax.bar(x + 0.2, [r["sklearn_quality"] for r in ok], 0.4, label="scikit-learn", color="#DD8452")
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("accuracy / R$^2$ / ARI (held-out)")
    ax.set_title("Predictive quality", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "benchmark_quality.png"), dpi=130)
    plt.close(fig)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Benchmark MLForge against scikit-learn")
    ap.add_argument("--quick", action="store_true", help="smaller data / fewer repeats")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="benchmarks/results")
    ap.add_argument("--only", nargs="*", help="substring filter on algorithm names")
    args = ap.parse_args(argv)
    rows = run_benchmarks(args.quick, args.repeats, args.only)
    save_results(rows, args.out)
    print(f"\nresults written to {os.path.abspath(args.out)}")
    return rows
