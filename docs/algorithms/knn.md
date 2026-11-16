# K-Nearest Neighbours

## Intuition
"Tell me who your neighbours are": predict by the labels (or values) of the $k$ closest training points. There is no training phase - the model *is* the data.

## Method
Distances come from `pairwise_distances` (Euclidean, Manhattan, Chebyshev, cosine, Minkowski-$p$). Euclidean distances use
$\|a-b\|^2=\|a\|^2+\|b\|^2-2a^\top b$ so the whole matrix is one BLAS matmul. Neighbours are ordered by a **stable** sort, so ties are broken
deterministically by training index (results never depend on memory layout).

* **Uniform voting:** majority class / mean value among the $k$ neighbours.
* **Distance weighting:** weights $w_i=1/d_i$; an exact match ($d=0$) dominates. `predict_proba` returns the weighted class shares.

**Choosing $k$.** Small $k$ -> jagged boundary, low bias / high variance; large $k$ -> smooth boundary, high bias. See `docs/images/knn_k_values.png`.
Features must be on comparable scales - standardise first.

## Complexity
Fit $O(1)$. Predict $O(n_{\text{train}}\,d)$ per query (brute force) plus a sort; memory $O(n_{\text{train}}d)$. Queries are processed in chunks to bound memory.

## Example
```python
from mlforge.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(k=7, weights="distance", metric="manhattan").fit(X, y)
dist, idx = knn.kneighbors(X_query)
```

## Verification
Class probabilities equal scikit-learn's to $10^{-9}$ on tie-free data for three metrics and both weightings; on data with ties (iris) the *distances* of the $k$ neighbours agree exactly (which of several equidistant points is returned is implementation-defined).

## References
Cover & Hart (1967) *Nearest neighbor pattern classification.*
