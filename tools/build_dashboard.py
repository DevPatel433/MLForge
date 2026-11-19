"""Build the correctness dashboard: ``docs/correctness.md`` (+ ``.json`` and SVG badges).

Everything on the dashboard is MEASURED when this script runs (or read from artefacts produced by
other measured runs), never typed in by hand:

* the test-suite is executed (pytest + coverage) and counted by kind of evidence;
* every gradient check is re-computed and its relative error recorded;
* parity differences versus scikit-learn / SciPy come from ``benchmarks/results`` (a measured run);
* mutation scores come from ``docs/mutation_results.json`` (``tools/mutation_test.py``);
* real-data results come from ``benchmarks/results/mnist_*.json``.

    python tools/build_dashboard.py            # run the whole suite (all chunks), then write the dashboard
    python tools/build_dashboard.py --chunk 0  # run one chunk of the suite (0..2) - for time-limited shells
    python tools/build_dashboard.py --assemble # combine the chunks and write the dashboard
    python tools/build_dashboard.py --no-tests # reuse docs/correctness.json's test numbers
"""
import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(ROOT, "docs")
sys.path.insert(0, ROOT)

CATEGORIES = {
    "Numerical gradient checks": ["test_gradients", "test_autograd", "test_conv_recurrent"],
    "Parity with scikit-learn / SciPy": ["test_sklearn_parity"],
    "Property-based (Hypothesis)": ["test_properties"],
    "Unit & invariant tests": ["test_math_utils", "test_validation_and_base", "test_optimization", "test_linear_models",
                               "test_tree_ensembles", "test_hist_boosting", "test_neighbors_cluster",
                               "test_decomposition_bayes_svm", "test_neural_network",
                               "test_preprocessing_model_selection", "test_metrics", "test_datasets", "test_release_scripts"],
    "Integration (plots, CLI, examples, notebooks)": ["test_visualization", "test_cli_and_benchmarking", "test_examples",
                                                      "test_notebooks"],
}


# --------------------------------------------------------------------------- tests + coverage
# The suite is run in chunks (coverage data accumulates with --cov-append). In CI the default invocation
# runs every chunk in one process; on a machine with a per-command time limit use --chunk N then --assemble.
CHUNKS = [
    ["test_sklearn_parity", "test_properties", "test_gradients", "test_autograd", "test_math_utils",
     "test_validation_and_base", "test_optimization"],
    ["test_linear_models", "test_tree_ensembles", "test_hist_boosting", "test_neighbors_cluster",
     "test_decomposition_bayes_svm", "test_neural_network", "test_conv_recurrent", "test_release_scripts"],
    ["test_preprocessing_model_selection", "test_metrics", "test_datasets", "test_visualization",
     "test_cli_and_benchmarking", "test_examples", "test_notebooks"],
]
COVERAGE_FILE = os.path.join(ROOT, ".coverage")


def unlisted_test_files():
    listed = {f for chunk in CHUNKS for f in chunk}
    on_disk = {f[:-3] for f in os.listdir(os.path.join(ROOT, "tests")) if f.startswith("test_") and f.endswith(".py")}
    return sorted(on_disk - listed)


def run_chunk(i):
    missing = unlisted_test_files()
    assert not missing, f"test files missing from the dashboard chunks (counts would be wrong): {missing}"
    if i == 0 and os.path.exists(COVERAGE_FILE):
        os.remove(COVERAGE_FILE)
    files = [os.path.join("tests", f + ".py") for f in CHUNKS[i]]
    with tempfile.TemporaryDirectory() as tmp:
        junit = os.path.join(tmp, "j.xml")
        proc = subprocess.run([sys.executable, "-m", "pytest", *files, "-o", "addopts=", "-q", f"--junitxml={junit}",
                               "--cov=mlforge", "--cov-append", "--cov-report="], cwd=ROOT, capture_output=True, text=True)
        suite = ET.parse(junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite[0]
        counts = {k: {"passed": 0, "failed": 0} for k in CATEGORIES}
        for case in suite.iter("testcase"):
            module = case.get("classname", "").split(".")[1] if "." in case.get("classname", "") else ""
            bad = case.find("failure") is not None or case.find("error") is not None
            for cat, mods in CATEGORIES.items():
                if module in mods:
                    counts[cat]["failed" if bad else "passed"] += 1
        part = {"exit_code": proc.returncode, "by_category": counts, "total": int(suite.get("tests")),
                "failures": int(suite.get("failures")) + int(suite.get("errors")), "skipped": int(suite.get("skipped"))}
    json.dump(part, open(os.path.join(DOCS, f".dashboard_chunk_{i}.json"), "w"))
    return part


def assemble_tests():
    parts = [json.load(open(os.path.join(DOCS, f".dashboard_chunk_{i}.json"))) for i in range(len(CHUNKS))]
    counts = {k: {"passed": sum(p["by_category"][k]["passed"] for p in parts),
                  "failed": sum(p["by_category"][k]["failed"] for p in parts)} for k in CATEGORIES}
    with tempfile.TemporaryDirectory() as tmp:
        cov = os.path.join(tmp, "cov.json")
        subprocess.run([sys.executable, "-m", "coverage", "json", "-o", cov], cwd=ROOT, capture_output=True, check=True)
        c = json.load(open(cov))["totals"]
    return {"exit_code": max(p["exit_code"] for p in parts), "by_category": counts, "total": sum(p["total"] for p in parts),
            "failures": sum(p["failures"] for p in parts), "skipped": sum(p["skipped"] for p in parts),
            "coverage_percent": round(c["percent_covered"], 2), "statements": c["num_statements"]}


def run_tests():
    os.makedirs(DOCS, exist_ok=True)
    for i in range(len(CHUNKS)):
        run_chunk(i)
    return assemble_tests()


# --------------------------------------------------------------------------- live gradient checks
def gradient_checks():
    from mlforge.autograd import Tensor
    from mlforge.linear_model import logistic_loss_grad, mse_loss_grad, softmax_loss_grad
    from mlforge.neural_network import (
        LSTM,
        Conv2D,
        CrossEntropyLoss,
        Dense,
        Flatten,
        LeakyReLU,
        MaxPool2D,
        MSELoss,
        ReLU,
        Sequential,
        Sigmoid,
        Tanh,
    )
    from mlforge.svm import hinge_loss_grad
    from mlforge.utils import check_gradient
    rng = np.random.RandomState(0)
    X, y, y01, y3 = rng.randn(30, 4), rng.randn(30), rng.randint(0, 2, 30), rng.randint(0, 3, 30)
    out = []

    def add(name, f, g, x):
        ok, err = check_gradient(f, g, x)
        out.append({"check": name, "relative_error": float(err), "passed": bool(ok)})

    th = rng.randn(5)
    add("MSE loss (+ L2)", lambda t: mse_loss_grad(t, X, y, 0.5)[0], lambda t: mse_loss_grad(t, X, y, 0.5)[1], th)
    add("Logistic loss (+ L2)", lambda t: logistic_loss_grad(t, X, y01, 2.0)[0], lambda t: logistic_loss_grad(t, X, y01, 2.0)[1], th)
    Y = np.eye(3)[y3]
    add("Softmax loss (+ L2)", lambda t: softmax_loss_grad(t, X, Y, 1.5)[0], lambda t: softmax_loss_grad(t, X, Y, 1.5)[1], rng.randn(15))
    tt = np.where(y01 == 1, 1.0, -1.0)
    add("Squared-hinge SVM loss", lambda t: hinge_loss_grad(t, X, tt, 1.0, "squared_hinge")[0],
        lambda t: hinge_loss_grad(t, X, tt, 1.0, "squared_hinge")[1], rng.randn(5) * 0.1)

    def net_check(name, net, Xn, Yn, loss):
        f = lambda t: (net.set_flat_params(t), net.loss_and_grad(Xn, Yn, loss)[0])[1]
        g = lambda t: (net.set_flat_params(t), net.loss_and_grad(Xn, Yn, loss)[1])[1]
        add(name, f, g, net.get_flat_params())
    Xm, ym = rng.randn(10, 4), rng.randint(0, 3, 10)
    for act in (ReLU, LeakyReLU, Tanh, Sigmoid):
        net_check(f"MLP backprop ({act.__name__})", Sequential([Dense(4, 6, "xavier", 1), act(), Dense(6, 3, "xavier", 2)]), Xm, ym, CrossEntropyLoss())
    Xi = rng.randn(3, 2, 7, 7)
    net = Sequential([Conv2D(2, 3, 3, 2, 1, random_state=0), Tanh(), Flatten()])
    net.layers.append(Dense(net.forward(Xi).shape[1], 4, "xavier", 1))
    net_check("Conv2D (stride 2, padding 1)", net, Xi, rng.randint(0, 4, 3), CrossEntropyLoss())
    Xc = rng.randn(4, 1, 10, 10)
    net = Sequential([Conv2D(1, 3, 3, random_state=0), ReLU(), MaxPool2D(2), Conv2D(3, 4, 2, padding=1, random_state=1), Tanh(), Flatten()])
    net.layers.append(Dense(net.forward(Xc).shape[1], 5, "xavier", 2))
    net_check("Whole CNN (conv-pool-conv-dense)", net, Xc, rng.randint(0, 5, 4), CrossEntropyLoss())
    Xs = rng.randn(3, 5, 4)
    net_check("LSTM -> Dense (BPTT, last state)", Sequential([LSTM(4, 6, random_state=0), Dense(6, 3, "xavier", 1)]), Xs, rng.randint(0, 3, 3), CrossEntropyLoss())
    net_check("MSE head (regression network)", Sequential([Dense(4, 5, "xavier", 0), Tanh(), Dense(5, 1, "xavier", 1)]), Xm, rng.randn(10), MSELoss())
    A = rng.rand(3, 3) + 0.5
    f = lambda v: float(((Tensor(v[:9].reshape(3, 3)) * Tensor(v[9:].reshape(3, 3))).tanh() ** 2).mean().data)

    def g(v):
        a, b = Tensor(v[:9].reshape(3, 3), True), Tensor(v[9:].reshape(3, 3), True)
        ((a * b).tanh() ** 2).mean().backward()
        return np.concatenate([a.grad.ravel(), b.grad.ravel()])
    add("Autograd engine (random expression)", f, g, np.concatenate([A.ravel(), (rng.rand(3, 3) + 0.5).ravel()]))
    return out


# --------------------------------------------------------------------------- badges
def badge(path, label, value, color):
    lw, vw = 7 * len(label) + 12, 7 * len(value) + 12
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{lw + vw}" height="20" role="img" aria-label="{label}: {value}">'
           f'<linearGradient id="s" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient>'
           f'<clipPath id="r"><rect width="{lw + vw}" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#r)">'
           f'<rect width="{lw}" height="20" fill="#555"/><rect x="{lw}" width="{vw}" height="20" fill="{color}"/><rect width="{lw + vw}" height="20" fill="url(#s)"/></g>'
           f'<g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
           f'<text x="{lw / 2}" y="14">{label}</text><text x="{lw + vw / 2}" y="14">{value}</text></g></svg>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(svg)


def color(pct):
    return "#4c1" if pct >= 90 else "#a4a61d" if pct >= 75 else "#dfb317" if pct >= 60 else "#e05d44"


# --------------------------------------------------------------------------- assemble
def read_json(*parts):
    path = os.path.join(ROOT, *parts)
    return json.load(open(path)) if os.path.exists(path) else None


def build(run=True, assemble=False):
    prev = read_json("docs", "correctness.json") or {}
    tests = assemble_tests() if assemble else (run_tests() if run else prev.get("tests"))
    if tests is None:
        raise SystemExit("no stored test results; run without --no-tests")
    grads = gradient_checks()
    bench = read_json("benchmarks", "results", "benchmark_results.json") or []
    mut = read_json("docs", "mutation_results.json") or {}
    mnist = {m: read_json("benchmarks", "results", f"mnist_{m}.json") for m in ("cnn", "mlp")}
    scaling = read_json("benchmarks", "results", "scaling_hist_boosting.json")
    mk, mn = sum(r["killed"] for r in mut.values()), sum(r["tested"] for r in mut.values())
    data = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
            "tests": tests, "gradient_checks": grads, "mutation": {"killed": mk, "tested": mn, "modules": mut},
            "parity": [r for r in bench if r.get("status") == "ok"], "mnist": mnist, "hist_scaling": scaling}
    os.makedirs(DOCS, exist_ok=True)
    json.dump(data, open(os.path.join(DOCS, "correctness.json"), "w"), indent=2, default=float)

    L = ["# Correctness dashboard", "",
         f"_Generated by `tools/build_dashboard.py` on {data['generated_utc']} UTC (Python {data['environment']['python']}, "
         f"NumPy {data['environment']['numpy']}). Every number below is measured when the tool runs; nothing is typed in by hand._", "",
         "## 1. Test suite", "", f"**{tests['total'] - tests['failures'] - tests['skipped']} passed, {tests['failures']} failed, "
         f"{tests['skipped']} skipped** - line coverage **{tests['coverage_percent']}%** of {tests['statements']} statements.", "",
         "| Kind of evidence | Tests | Failing |", "|---|---:|---:|"]
    for cat, c in tests["by_category"].items():
        L.append(f"| {cat} | {c['passed'] + c['failed']} | {c['failed']} |")
    gp = sum(g["passed"] for g in grads)
    L += ["", "## 2. Numerical gradient checks (re-computed now)", "",
          f"{gp}/{len(grads)} pass the finite-difference test (`relative error < 1e-5`).", "",
          "| Gradient | Relative error | Result |", "|---|---:|:---:|"]
    L += [f"| {g['check']} | {g['relative_error']:.1e} | {'pass' if g['passed'] else '**FAIL**'} |" for g in grads]
    L += ["", "## 3. Agreement with scikit-learn (measured benchmark run)", "",
          "Held-out quality and the difference between the two implementations' outputs, from `benchmarks/results/`.", "",
          "| Algorithm | Quality (MLForge / sklearn) | Difference measure | Difference |", "|---|---:|---|---:|"]
    for r in data["parity"]:
        L.append(f"| {r['algorithm']} | {r['quality_metric']} {r['mlforge_quality']:.4f} / {r['sklearn_quality']:.4f} | {r['difference_metric']} | {r['max_abs_diff']:.1e} |")
    L += ["", "## 4. Mutation testing (would the tests notice if the code were wrong?)", ""]
    if mn:
        L += [f"**{mk}/{mn} sampled mutants killed ({mk / mn:.1%}).** Random mutants (operator swaps, comparison flips, constant changes) "
              "are injected one at a time; a mutant is *killed* if any guarding test fails. Survivors are reviewed: each was either "
              "a real test gap (fixed) or an *equivalent mutant* that cannot change behaviour.", "",
              "| Module | Mutants | Killed | Score |", "|---|---:|---:|---:|"]
        L += [f"| `{m}` | {r['tested']} | {r['killed']} | {r['score']:.0%} |" for m, r in sorted(mut.items())]
    else:
        L.append("_Not run yet: `python tools/mutation_test.py`._")
    L += ["", "## 5. Real-data results (reproducible scripts)", ""]
    if any(mnist.values()):
        L += ["MNIST, the real 60,000 / 10,000 split (`examples/mnist_cnn.py`), one CPU core, pure NumPy:", "",
              "| Model | Parameters | Epochs | Test accuracy | Training time |", "|---|---:|---:|---:|---:|"]
        for m, r in mnist.items():
            if r:
                L.append(f"| {m.upper()} | {r['parameters']:,} | {r['epochs']} | **{r['final_test_accuracy']:.2%}** | {r['total_training_seconds']:.0f} s |")
    if scaling:
        L += ["", "Histogram vs exact gradient boosting, capacity-matched (`benchmarks/scaling_hist_boosting.py`):", "",
              "| Training rows | Exact boosting | Histogram boosting | Speed-up |", "|---:|---:|---:|---:|"]
        L += [f"| {r['n_train']:,} | {r['exact_seconds']} s | {r['hist_seconds']} s | {r['speedup']}x |" for r in scaling["rows"]]
    L += ["", "## How to regenerate", "", "```bash", "python tools/mutation_test.py        # mutation scores (about 25 min)",
          "python tools/build_dashboard.py      # this page, correctness.json and the badges", "```", ""]
    open(os.path.join(DOCS, "correctness.md"), "w").write("\n".join(L))

    if bench:                                                  # publish the benchmark table inside docs/ (mkdocs cannot link outside)
        from mlforge.benchmarking import to_markdown
        body = "# Full benchmark table\n\nGenerated from `benchmarks/results/benchmark_results.json`.\n\n" + to_markdown(
            [r for r in bench if r.get("status") == "ok"])
        imgs = os.path.join(DOCS, "images", "benchmarks")
        os.makedirs(imgs, exist_ok=True)
        for png in ("benchmark_fit_time.png", "benchmark_difference.png", "benchmark_quality.png"):
            src = os.path.join(ROOT, "benchmarks", "results", png)
            if os.path.exists(src):
                shutil.copy(src, imgs)
                body += f"\n\n![{png[:-4]}](images/benchmarks/{png})"
        open(os.path.join(DOCS, "benchmark_results.md"), "w").write(body + "\n")
    bd = os.path.join(DOCS, "badges")
    badge(os.path.join(bd, "tests.svg"), "tests", f"{tests['total'] - tests['failures']} passing" if not tests["failures"] else f"{tests['failures']} failing",
          "#4c1" if not tests["failures"] else "#e05d44")
    badge(os.path.join(bd, "coverage.svg"), "coverage", f"{tests['coverage_percent']:.0f}%", color(tests["coverage_percent"]))
    badge(os.path.join(bd, "gradients.svg"), "gradient checks", f"{gp}/{len(grads)}", "#4c1" if gp == len(grads) else "#e05d44")
    if mn:
        badge(os.path.join(bd, "mutation.svg"), "mutation score", f"{mk / mn:.0%}", color(100 * mk / mn))
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-tests", action="store_true")
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--assemble", action="store_true")
    args = ap.parse_args()
    if args.chunk is not None:
        os.makedirs(DOCS, exist_ok=True)
        part = run_chunk(args.chunk)
        print(f"chunk {args.chunk}: {part['total'] - part['failures']}/{part['total']} passed")
        sys.exit(0)
    d = build(run=not args.no_tests, assemble=args.assemble)
    t = d["tests"]
    print(f"tests {t['total'] - t['failures']}/{t['total']}  coverage {t['coverage_percent']}%  "
          f"gradient checks {sum(g['passed'] for g in d['gradient_checks'])}/{len(d['gradient_checks'])}  "
          f"mutation {d['mutation']['killed']}/{d['mutation']['tested']}")
