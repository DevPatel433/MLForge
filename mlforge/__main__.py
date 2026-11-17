"""Command-line interface:  ``python -m mlforge <command>``  (or ``mlforge <command>``)."""
import argparse
import importlib
import inspect
import sys

from . import __version__

FAMILIES = ["linear_model", "tree", "neighbors", "cluster", "decomposition", "bayes", "svm",
            "neural_network", "preprocessing", "model_selection"]


def list_estimators():
    """Return ``{family: [estimator names]}`` for every public estimator class."""
    from .base import BaseEstimator
    found = {}
    for fam in FAMILIES:
        mod = importlib.import_module(f"mlforge.{fam}")
        names = [n for n in getattr(mod, "__all__", []) if inspect.isclass(getattr(mod, n))
                 and issubclass(getattr(mod, n), BaseEstimator)]
        found[fam] = sorted(names)
    return found


def cmd_info(args):
    import numpy
    print(f"MLForge {__version__}   (NumPy {numpy.__version__})")
    est = list_estimators()
    total = sum(len(v) for v in est.values())
    print(f"{total} estimators across {len(est)} modules:\n")
    for fam, names in est.items():
        print(f"  {fam:<16} {', '.join(names)}")
    return 0


def cmd_demo(args):
    """Train a few models on a bundled-free synthetic dataset and print a metrics report."""
    from .datasets import make_blobs
    from .metrics import classification_report
    from .pipeline import Pipeline
    from .preprocessing import StandardScaler, train_test_split
    from .svm import SVC
    from .tree import RandomForestClassifier
    X, y = make_blobs(600, [[-2, -1], [2, 1], [0, 3]], cluster_std=1.3, random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    models = {
        "Pipeline(StandardScaler -> SVC rbf)": Pipeline([("sc", StandardScaler()), ("svc", SVC(random_state=0))]),
        "RandomForestClassifier": RandomForestClassifier(40, random_state=0),
    }
    for name, m in models.items():
        m.fit(Xtr, ytr)
        print(f"\n== {name}   test accuracy = {m.score(Xte, yte):.3f}")
        print(classification_report(yte, m.predict(Xte)))
    return 0


def cmd_benchmark(args):
    from .benchmarking import run_benchmarks, save_results, to_markdown
    rows = run_benchmarks(args.quick, args.repeats, args.only)
    save_results(rows, args.out, plots=not args.no_plots)
    print("\n" + to_markdown(rows))
    print(f"results written to {args.out}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


def build_parser():
    p = argparse.ArgumentParser(prog="mlforge", description="MLForge - machine learning from scratch")
    p.add_argument("--version", action="version", version=f"mlforge {__version__}")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("info", help="list the available estimators").set_defaults(func=cmd_info)
    sub.add_parser("demo", help="train and evaluate a few models on synthetic data").set_defaults(func=cmd_demo)
    b = sub.add_parser("benchmark", help="compare every algorithm with scikit-learn")
    b.add_argument("--quick", action="store_true", help="smaller data, fewer repeats")
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--out", default="benchmarks/results")
    b.add_argument("--only", nargs="*", help="substring filter on algorithm names")
    b.add_argument("--no-plots", action="store_true")
    b.set_defaults(func=cmd_benchmark)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
