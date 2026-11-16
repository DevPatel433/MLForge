# Benchmarks

`python -m mlforge benchmark` (or `python benchmarks/run_benchmarks.py`) compares **27 algorithms** with their scikit-learn equivalents and writes
`benchmarks/results/benchmark_results.{md,csv,json}` plus charts. Metrics per algorithm:

* **training time** and **prediction time** (median of 3 repeats; 1 for the slowest), single-threaded NumPy;
* **peak memory** during `fit` (`tracemalloc`; note that scikit-learn's compiled code allocates outside Python's tracker, so its memory column is a lower bound);
* **quality** on held-out data (accuracy / $R^2$ / adjusted Rand index / explained variance);
* **numerical difference** between the two implementations' outputs: max absolute difference of predictions or probabilities, label-disagreement rate, or $1-\text{ARI}$ for clusterings.

Full results table: [benchmark_results.md](benchmark_results.md) (also `benchmarks/results/` in the repository: markdown, CSV, JSON and charts).

## What the numbers say
* **Deterministic, convex problems agree to machine precision**: OLS, Ridge, Lasso, ElasticNet, PCA, Gaussian NB, KNN, K-Means, hierarchical clustering, SVC with the RBF kernel (identical predictions), logistic regression via Newton (probabilities within $4\times10^{-7}$).
* **Iterative first-order solvers match once converged** (logistic regression by gradient descent: $7\times10^{-7}$) but are 80-120x slower than scikit-learn's L-BFGS - a Python loop over epochs versus compiled code. That is the cost of being readable.
* **Randomised / greedy ensembles are statistically, not bit-wise, equivalent**: random forests, boosting, MLPs and the perceptron legitimately differ from scikit-learn in individual predictions (different random streams, different tie-breaking between equally good splits) while matching in held-out quality. (Boosting uses plain variance reduction where scikit-learn uses Friedman's MSE criterion, hence its large *maximum* probability difference at equal accuracy.)
* **Tree learning and SMO** are the slowest relative to scikit-learn that is not a Python-loop solver (1.8-14x for trees / boosting, 11x for SVC), since each node is a handful of vectorised NumPy calls rather than compiled loops.

Scaling the benchmark: `--quick` shrinks data and repeats; `--only Ridge KMeans` filters by name.
