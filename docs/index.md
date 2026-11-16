# MLForge documentation

| Section | Contents |
|---|---|
| [API reference](api.md) | every public class and function (auto-generated) |
| [Benchmarks](benchmarks.md) | MLForge vs scikit-learn: time, memory, quality, numerical difference |
| [Optimization](algorithms/optimization.md) | GD / momentum / Nesterov / AdaGrad / RMSProp / Adam, schedules |
| [Linear regression](algorithms/linear_regression.md) | OLS, normal equations, gradient descent, polynomial features |
| [Regularisation](algorithms/regularization.md) | Ridge, Lasso (coordinate descent, ISTA), Elastic Net |
| [Logistic regression](algorithms/logistic_regression.md) | binary, softmax, Newton / IRLS, perceptron |
| [Decision trees](algorithms/decision_trees.md) | CART, Gini / entropy, cost-complexity pruning |
| [Ensembles](algorithms/ensembles.md) | random forests (OOB), gradient boosting |
| [K-nearest neighbours](algorithms/knn.md) | distances, weighting, deterministic ties |
| [Naive Bayes](algorithms/naive_bayes.md) | Gaussian, multinomial, Bernoulli |
| [Clustering](algorithms/kmeans.md) | K-Means++, hierarchical, DBSCAN, Gaussian mixtures (EM) |
| [PCA](algorithms/pca.md) | covariance / SVD derivations, explained variance |
| [SVM](algorithms/svm.md) | hinge loss, dual, kernels, SMO |
| [Neural networks](algorithms/neural_networks.md) | backprop derivation, initialisation, dropout |
| [Autograd](algorithms/autograd.md) | the reverse-mode autodiff engine |
| [Metrics](algorithms/metrics.md) | classification, regression, clustering metrics |
| [Model selection](algorithms/model_selection.md) | CV, grid search, pipelines, estimator protocol |

Every algorithm page has the same structure: intuition, derivation, algorithm, complexity, example, **how it is verified**, references.

## How correctness is established
1. **Numerical gradient checks** - analytic gradients of every gradient-based model are compared with finite differences (`tests/test_gradients.py`).
2. **Reference parity** - outputs are compared with scikit-learn / SciPy at tight tolerances (`tests/test_sklearn_parity.py`); where randomness or tie-breaking makes bit-equality meaningless, the tests assert structural equivalents (leaf counts, neighbour *distances*, ARI).
3. **Mathematical invariants** - EM never decreases the likelihood, SMO satisfies the dual constraints, PCA components are orthonormal, K-Means inertia is monotone, folds partition the data...
4. **Hand-computed examples** - Naive Bayes posteriors, metric values, optimizer first steps.
5. **Edge cases and invalid input** - NaNs, wrong shapes, unseen categories, constant features, single-class data, unfitted models.

Gallery (`examples/make_gallery.py` regenerates every image): see the README.
