# Evaluation Metrics

All metrics are validated against scikit-learn on random data (`tests/test_sklearn_parity.py`).

## Classification
With confusion-matrix entries TP, FP, FN, TN:
$$\text{accuracy}=\tfrac{TP+TN}{n},\ \ \text{precision}=\tfrac{TP}{TP+FP},\ \ \text{recall}=\tfrac{TP}{TP+FN},\ \ F_\beta=\tfrac{(1+\beta^2)PR}{\beta^2P+R}.$$
Multiclass: per-class values combined by `average` = `macro` (unweighted mean), `micro` (pooled counts; equals accuracy for single-label problems) or `weighted` (by support).
Also: `balanced_accuracy_score`, `cohen_kappa_score`, `matthews_corrcoef`, `classification_report`.

* **ROC / AUC:** `roc_curve` sweeps the decision threshold; `auc` is the trapezoidal area. AUC equals the probability that a random positive outranks a random negative (ties count one half) and is 0.5 for uninformative scores. Multiclass AUC is one-vs-rest macro.
* **Precision-recall:** `precision_recall_curve`, `average_precision_score` $=\sum_n(R_n-R_{n-1})P_n$ - preferable to ROC for heavily imbalanced data.
* **Log loss / Brier:** $-\tfrac1n\sum\log p_{i,y_i}$ rewards calibrated probabilities; Brier is the mean squared probability error.

## Regression
MAE, MSE, RMSE, $R^2=1-\frac{SS_{res}}{SS_{tot}}$ (negative when worse than predicting the mean), MAPE, median absolute error, max error, explained variance.

## Clustering
* **Silhouette** $s_i=\frac{b_i-a_i}{\max(a_i,b_i)}\in[-1,1]$ (higher is better; singletons score 0); `silhouette_samples` for per-point values.
* **Davies-Bouldin** (lower is better), **Calinski-Harabasz** (higher is better).
* **Adjusted Rand index** and **normalised mutual information** compare two labelings and are invariant to label permutation; ARI is $\approx0$ for random labelings.

## Edge cases handled
Division by zero returns 0 (precision of a class never predicted), single-class AUC raises a clear error, clipping avoids `log(0)` in log loss, `average='binary'` on a multiclass target raises instead of silently mis-scoring.
