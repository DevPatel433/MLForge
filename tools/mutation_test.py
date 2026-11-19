"""A small, dependency-free mutation-testing tool.

Mutation testing answers "would my tests notice if the code were wrong?".
It makes ONE small change at a time to the source (``+`` -> ``-``, ``<`` -> ``<=``,
``and`` -> ``or``, a constant off by one, ...), runs the tests that guard that module, and
records whether any test failed (the mutant is *killed*) or all passed (it *survived*,
exposing either a test gap or an *equivalent* mutation that cannot change behaviour).

    python tools/mutation_test.py                         # every configured module
    python tools/mutation_test.py metrics/ --max 20       # modules matching a substring
    python tools/mutation_test.py --report                # stored summary table
    python tools/mutation_test.py --survivors metrics/    # list surviving mutants with source

The baseline (unmutated) tests must pass first - otherwise every mutant would look "killed".
Results are written to ``docs/mutation_results.json``. Mutants are sampled at random per module
(fixed ``--seed``), so the score is an estimate, not an exhaustive census.
"""
import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "docs", "mutation_results.json")
P = "tests/test_sklearn_parity.py::"

# module (relative to mlforge/)  ->  the tests that guard it (files or node ids)
TARGETS = {
    "utils/math_utils.py": ["tests/test_math_utils.py", "tests/test_properties.py"],
    "utils/pairwise.py": ["tests/test_math_utils.py", "tests/test_properties.py"],
    "utils/gradcheck.py": ["tests/test_gradients.py"],
    "metrics/classification.py": ["tests/test_metrics.py", "tests/test_properties.py", P + "test_all_metrics_match"],
    "metrics/regression.py": ["tests/test_metrics.py", "tests/test_properties.py", P + "test_all_metrics_match"],
    "metrics/cluster.py": ["tests/test_metrics.py", "tests/test_properties.py", P + "test_all_metrics_match"],
    "optimization/optimizers.py": ["tests/test_optimization.py", "tests/test_properties.py"],
    "optimization/schedules.py": ["tests/test_optimization.py"],
    "optimization/optimize.py": ["tests/test_optimization.py"],
    "linear_model/losses.py": ["tests/test_gradients.py"],
    "linear_model/regularized.py": ["tests/test_linear_models.py", P + "test_ols_ridge_lasso_elasticnet"],
    "preprocessing/scalers.py": ["tests/test_preprocessing_model_selection.py", "tests/test_properties.py", P + "test_scalers_and_encoders_match"],
    "preprocessing/encoders.py": ["tests/test_preprocessing_model_selection.py", "tests/test_properties.py", P + "test_scalers_and_encoders_match"],
    "model_selection/split.py": ["tests/test_preprocessing_model_selection.py", "tests/test_properties.py"],
    "decomposition/pca.py": ["tests/test_decomposition_bayes_svm.py", "tests/test_properties.py", P + "test_pca_matches"],
    "bayes/naive_bayes.py": ["tests/test_decomposition_bayes_svm.py", P + "test_naive_bayes_matches"],
    "neural_network/layers.py": ["tests/test_neural_network.py", "tests/test_gradients.py"],
    "autograd/tensor.py": ["tests/test_autograd.py"],
    "neural_network/conv.py": ["tests/test_conv_recurrent.py"],
    "neural_network/recurrent.py": ["tests/test_conv_recurrent.py"],
    "tree/hist_boosting.py": ["tests/test_hist_boosting.py"],
    "tree/decision_tree.py": ["tests/test_tree_ensembles.py", P + "test_decision_tree_structure_matches"],
}

_SWAPS = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult, ast.Pow: ast.Mult}
_CMPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}


class _Collector(ast.NodeVisitor):
    """Find every mutable site; lines marked ``# pragma: no mutate`` are skipped."""

    def __init__(self, lines):
        self.sites, self.lines = [], lines

    def _skip(self, node):
        return "pragma: no mutate" in self.lines[node.lineno - 1]

    def visit_Expr(self, node):                       # docstrings / bare strings
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        self.generic_visit(node)

    def visit_BinOp(self, node):
        if type(node.op) in _SWAPS and not self._skip(node):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) == 1 and type(node.ops[0]) in _CMPS and not self._skip(node):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        if not self._skip(node):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not) and not self._skip(node):
            self.sites.append(node)
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool) and not self._skip(node):
            self.sites.append(node)


def _normalised(source):
    """Canonical source text: all line numbers in this tool refer to this form."""
    return ast.unparse(ast.parse(source))


def _sites(tree, text):
    c = _Collector(text.split("\n"))
    c.visit(tree)
    return c.sites


def _mutate(text, target_index):
    tree = ast.parse(text)
    node = _sites(tree, text)[target_index]
    if isinstance(node, ast.BinOp):
        desc = f"{type(node.op).__name__} -> {_SWAPS[type(node.op)].__name__}"
        node.op = _SWAPS[type(node.op)]()
    elif isinstance(node, ast.Compare):
        desc = f"{type(node.ops[0]).__name__} -> {_CMPS[type(node.ops[0])].__name__}"
        node.ops[0] = _CMPS[type(node.ops[0])]()
    elif isinstance(node, ast.BoolOp):
        desc = "and <-> or"
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif isinstance(node, ast.UnaryOp):
        desc = "remove `not`"
        node.op = ast.UAdd()
    else:
        old = node.value
        node.value = (old + 1) if old not in (0, 0.0) else 1
        desc = f"constant {old!r} -> {node.value!r}"
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), f"line {node.lineno}: {desc}"


def _pytest(tests, work, env, timeout):
    return subprocess.run([sys.executable, "-m", "pytest", *tests, "-x", "-q", "-o", "addopts=", "-p", "no:cacheprovider"],
                          cwd=work, env=env, capture_output=True, text=True, timeout=timeout)


def run_module(rel, tests, max_mutants, seed, timeout=180, verbose=True):
    text = _normalised(open(os.path.join(ROOT, "mlforge", rel)).read())
    n_sites = len(_sites(ast.parse(text), text))
    picks = sorted(random.Random(seed).sample(range(n_sites), min(max_mutants, n_sites)))
    work = tempfile.mkdtemp(prefix="mut_")
    shutil.copytree(os.path.join(ROOT, "mlforge"), os.path.join(work, "mlforge"))
    shutil.copytree(os.path.join(ROOT, "tests"), os.path.join(work, "tests"))
    shutil.copy(os.path.join(ROOT, "pyproject.toml"), work)
    target = os.path.join(work, "mlforge", rel)
    env = dict(os.environ, PYTHONPATH=work, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    results = []
    try:
        base = _pytest(tests, work, env, timeout * 2)           # a mutant is only "killed" if the BASELINE passes
        if base.returncode != 0:
            raise RuntimeError(f"baseline tests fail for {rel}; refusing to mutate:\n{base.stdout[-600:]}")
        for i, idx in enumerate(picks, 1):
            mutated, desc = _mutate(text, idx)
            open(target, "w").write(mutated)
            t0 = time.time()
            try:
                killed = _pytest(tests, work, env, timeout).returncode != 0
            except subprocess.TimeoutExpired:
                killed = True                                   # an infinite loop is also detected
            results.append({"site": idx, "mutation": desc, "killed": killed, "seconds": round(time.time() - t0, 2)})
            if verbose:
                print(f"  [{i:>3}/{len(picks)}] {'killed  ' if killed else 'SURVIVED'} {desc}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    killed = sum(r["killed"] for r in results)
    return {"module": rel, "sites": n_sites, "tested": len(results), "killed": killed,
            "score": round(killed / max(len(results), 1), 4), "tests": tests, "mutants": results}


def load():
    return json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}


def save(store):
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    json.dump(store, open(RESULTS, "w"), indent=2)


def report(store):
    rows = sorted(store.values(), key=lambda r: r["module"])
    k, n = sum(r["killed"] for r in rows), sum(r["tested"] for r in rows)
    print(f"{'module':<34}{'tested':>7}{'killed':>8}{'score':>8}")
    for r in rows:
        print(f"{r['module']:<34}{r['tested']:>7}{r['killed']:>8}{r['score']:>8.1%}")
    if n:
        print(f"{'TOTAL':<34}{n:>7}{k:>8}{k / n:>8.1%}")
    return k, n


def survivors(store, filters):
    for rel, r in sorted(store.items()):
        if filters and not any(f in rel for f in filters):
            continue
        lines = _normalised(open(os.path.join(ROOT, "mlforge", rel)).read()).split("\n")
        for m in r["mutants"]:
            if not m["killed"]:
                n = int(re.search(r"line (\d+)", m["mutation"]).group(1))
                print(f"{rel:<30} {m['mutation']:<34} | {lines[n - 1].strip()[:80]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="*", help="substring filters, e.g. metrics/classification")
    ap.add_argument("--max", type=int, default=12, help="mutants sampled per module")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--survivors", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    store = load()
    if args.report:
        report(store)
        return 0
    if args.survivors:
        survivors(store, args.modules)
        return 0
    for rel, tests in TARGETS.items():
        if args.modules and not any(m in rel for m in args.modules):
            continue
        print(f"== {rel}", flush=True)
        store[rel] = run_module(rel, tests, args.max, args.seed, verbose=not args.quiet)
        save(store)
        print(f"   score {store[rel]['score']:.1%} ({store[rel]['killed']}/{store[rel]['tested']})", flush=True)
    report(store)
    return 0


if __name__ == "__main__":
    sys.exit(main())
