# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/); versioning: [SemVer](https://semver.org/).

## [0.3.0]

Focus: evidence, scale and release readiness.

### Added
- **Property-based tests** (21 Hypothesis laws) and a dependency-free **mutation-testing tool** (`tools/mutation_test.py`; 207 / 220 sampled mutants killed, survivors reviewed in `docs/mutation_survivors.md`).
- **Correctness dashboard** (`tools/build_dashboard.py`): measured test / coverage / gradient-check / parity / mutation / real-data figures, SVG badges, CI workflow.
- **Convolution and pooling layers** (`Conv2D` with stride and padding via im2col, `MaxPool2D`, `Flatten`) and an **`LSTM`** layer with full BPTT, each verified by finite-difference gradient checks and against naive reference implementations.
- **Real MNIST**: `load_mnist` (SHA-256 verified, cached) and `examples/mnist_cnn.py`; a 5,258-parameter CNN reaches 98.57% test accuracy, an MLP 97.82%; saved weights and JSON logs; figures via `examples/mnist_report.py`.
- **Histogram gradient boosting** (`HistGradientBoostingRegressor` / `Classifier`): binning, histogram subtraction, leaf-wise growth; 2.3x-12.6x faster than exact boosting from 2k to 32k rows (`benchmarks/scaling_hist_boosting.py`).
- Release preparation: MkDocs site (strict build) + GitHub Pages workflow, `docs/RELEASING.md`, `.zenodo.json`, JOSS-format `paper.md` / `paper.bib`, an essay on verifying ML code, `scripts/set_repo_url.py`, `scripts/rename_package.py`, workflows for docs, dashboard and weekly mutation testing.
- Benchmark suite now covers 29 algorithms.

### Fixed (found by the new tests)
- `StandardScaler` returned +-1 instead of 0 for constant columns whose floating-point mean was inexact.
- **Infinite recursion** in tree growth when two feature values were one floating-point step apart (the midpoint rounded up to the larger value).
- `Sequential.summary()` mislabelled convolution layers.
- Many test gaps closed after mutation testing (e.g. `recall_score` could have returned F1 undetected; ReLU's backward pass was only tested on positive inputs).

### Changed
- PyPI distribution name is `mlforge-numpy` (`mlforge` is taken on PyPI); the import name is unchanged - see `docs/RELEASING.md` for the namespace caveat and the rename script.

## [0.2.0]

A large release: the library grew from 12 to 38 estimators and gained a verification layer.

### Added
- **Verification layer**: `mlforge.utils.check_gradient` / `numerical_gradient`; finite-difference tests for every gradient-based model and whole neural networks; a scikit-learn / SciPy parity suite; mathematical-invariant tests.
- **`mlforge.autograd`**: reverse-mode automatic differentiation (`Tensor`, `no_grad`).
- **Estimator protocol** (`BaseEstimator`): `get_params` / `set_params` / `clone` / `save` / `load`, readable `repr`, automatic input validation (helpful errors), `NotFittedError`.
- **Optimization**: AdaGrad, learning-rate schedules (step, exponential, inverse-time, cosine, warm-up), a shared mini-batch training loop with early stopping, `minimize()` returning full trajectories, Rosenbrock / Beale test surfaces. Optimizer state is now reset on every `fit`.
- **Linear models**: closed-form and gradient-descent solvers; Ridge closed form; Lasso / ElasticNet by coordinate descent and ISTA; Newton / IRLS logistic regression with L2 (`C`); `Perceptron`; multi-feature `PolynomialRegression` with interaction terms.
- **Trees**: vectorised CART, `DecisionTreeRegressor`, cost-complexity pruning, `feature_importances_`, `export_text`, `predict_proba`, `RandomForestRegressor`, OOB scores, `GradientBoostingClassifier` / `Regressor`.
- **SVM**: kernel `SVC` trained with Platt's SMO (linear / poly / RBF / sigmoid / custom kernels, one-vs-one multiclass); `LinearSVC` with hinge / squared hinge and one-vs-rest.
- **Neural networks**: `Sequential`, `Dropout`, LeakyReLU, three losses, per-parameter optimizers, L2, early stopping, `MLPRegressor`, `history_`, model `summary()`.
- **Clustering**: hierarchical clustering (4 linkages, SciPy-compatible linkage matrix), DBSCAN, Gaussian mixtures (EM, BIC / AIC, sampling), K-Means restarts and empty-cluster handling, `elbow_curve`, `silhouette_curve`.
- **Other models**: Bernoulli NB, PCA with SVD solver / whitening / variance-target components, k-NN with 5 metrics and `kneighbors`.
- **Preprocessing**: MaxAbs / Robust scalers, Normalizer, PolynomialFeatures, OrdinalEncoder, multi-column OneHotEncoder, KNNImputer, `constant` imputation.
- **Model selection**: stratified split, KFold / StratifiedKFold / ShuffleSplit, `cross_val_score`, `cross_val_predict`, `GridSearchCV`, `RandomizedSearchCV`, `learning_curve`, `validation_curve`, `Pipeline`.
- **Metrics**: 32 functions incl. multiclass averaging, ROC / PR curves, MCC, kappa, log loss, Brier, silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI.
- **Visualization** (25 functions): decision boundaries, ROC / PR, confusion matrices, trees, dendrograms, SVM margins, regularisation paths, optimizer races, animated GIFs.
- `mlforge.datasets` (NumPy-only generators), `mlforge.benchmarking` + CLI (`python -m mlforge info | demo | benchmark`), six executed notebooks, per-algorithm documentation with derivations, generated API reference, Makefile, CI.

### Changed (breaking relative to 0.1.0)
- Fitted attributes follow the `coef_` / `intercept_` / `cost_history_` convention (the old names remain as read-only aliases).
- `predict_proba` of binary classifiers returns shape `(n, 2)`.
- `precision_score` / `recall_score` / `f1_score` take `pos_label` and `average` (was `positive_label`).
- `KMeans(n_init=5)` by default; `PCA` components are stored as `components_` of shape `(k, d)`.
- Model constructors are no longer positional-compatible where hyper-parameters were reordered; use keywords.

### Fixed
- Nesterov momentum used a non-standard update (and a dead code branch).
- Learning-rate schedules were off by one step.
- Zero-gain splits were rejected, so trees could not learn XOR.
- `Normalizer.fit_transform` did not mark the transformer as fitted.
- k-NN tie-breaking depended on memory layout; it is now deterministic (lowest training index).
- Optimizer state leaked between successive `fit` calls.

## [0.1.0]
- Initial scaffold: math utilities, optimizers, linear / logistic / softmax regression, decision tree, random forest, k-NN, K-Means, PCA, Naive Bayes, linear SVM, MLP, preprocessing, metrics, 25 tests, CI.
