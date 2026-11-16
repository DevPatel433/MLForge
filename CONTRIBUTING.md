# Contributing to MLForge

Thanks for helping! MLForge is a teaching-quality, *verified* from-scratch library, so correctness evidence matters as much as code.

## Setup

```bash
git clone https://github.com/your-username/mlforge.git && cd mlforge
make install        # pip install -e ".[dev]"
make lint           # ruff
make cov            # tests + coverage (must stay >= 95%)
```

## Ground rules

1. **NumPy only inside `mlforge/`.** scikit-learn / SciPy may appear only in `mlforge/benchmarking.py`, `tests/`, `examples/`, `notebooks/` (and there only for benchmarking, parity checks, or *loading* toy datasets). A test enforces this.
2. **Every algorithm ships with evidence**:
   * unit tests on small deterministic problems *and* edge cases / invalid input;
   * a **gradient check** (`mlforge.utils.check_gradient`) if it has a hand-derived gradient;
   * a **parity test** against scikit-learn / SciPy where the maths is deterministic - or an invariant test (likelihood monotone, constraints satisfied, ...) where it is not;
   * a docs page in `docs/algorithms/` with intuition, derivation, complexity, example and a "Verification" section.
3. **Follow the estimator protocol**: subclass `BaseEstimator` (+ `ClassifierMixin` / `RegressorMixin`), put hyper-parameters in `__init__` *only* (stored under the same name), learned state in attributes ending with `_`, return `self` from `fit`. Validation and `NotFittedError` are automatic.
4. **Readability first.** Prefer clear NumPy over clever tricks; comment the maths (cite the equation), not the syntax.
5. **Measure your tests.** Run `python tools/mutation_test.py <module>` on code you add; a surviving mutant is either a test gap (fix it) or an equivalent mutant (document it in `docs/mutation_survivors.md`). Property-based tests (`tests/test_properties.py`) are encouraged for numerical edge cases.
6. **Honest benchmarks.** If you add an algorithm, add a case to `mlforge/benchmarking.py`. Do not tune the reference implementation down.

## Pull requests

* One logical change per PR; describe *why*, link the issue.
* Run `make all` (lint + coverage + regenerate API docs) before pushing; regenerate notebooks / gallery if you changed behaviour they show (`make notebooks`, `make gallery`).
* Update `CHANGELOG.md`.

## Reporting bugs

Include a minimal reproducible example (data + code), what you expected, and what happened. A failing test is the best bug report.

## Code of conduct

Be kind and constructive - see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
