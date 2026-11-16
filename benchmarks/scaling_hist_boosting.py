"""How does histogram boosting scale with the number of samples, compared with exact-split boosting?

Capacity-matched: exact trees with max_depth=5 (<= 32 leaves) vs histogram trees with 31 leaves,
20 boosting iterations, 20 features. Results -> benchmarks/results/scaling_hist_boosting.json

    python benchmarks/scaling_hist_boosting.py
"""
import json
import os
import time

import numpy as np

from mlforge.tree import GradientBoostingRegressor, HistGradientBoostingRegressor

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "scaling_hist_boosting.json")


def timed(make, X, y):
    t = time.perf_counter()
    m = make().fit(X, y)
    return time.perf_counter() - t, m


def main(sizes=(2000, 8000, 32000), iters=20):
    rng = np.random.RandomState(0)
    w = rng.randn(20)
    rows = []
    for n in sizes:
        X = rng.randn(n + 2000, 20)
        y = X @ w + 2 * np.sin(2 * X[:, 0]) + rng.randn(n + 2000) * 0.5
        tr, te = slice(0, n), slice(n, None)
        t_exact, exact = timed(lambda: GradientBoostingRegressor(iters, max_depth=5, random_state=0), X[tr], y[tr])
        t_hist, hist = timed(lambda: HistGradientBoostingRegressor(iters, max_leaf_nodes=31), X[tr], y[tr])
        row = {"n_train": n, "exact_seconds": round(t_exact, 2), "hist_seconds": round(t_hist, 2),
               "speedup": round(t_exact / t_hist, 1), "exact_r2": round(exact.score(X[te], y[te]), 4),
               "hist_r2": round(hist.score(X[te], y[te]), 4)}
        rows.append(row)
        print(row, flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"iterations": iters, "features": 20, "rows": rows}, open(OUT, "w"), indent=2)
    return rows


if __name__ == "__main__":
    main()
