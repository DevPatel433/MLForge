# Clustering: K-Means, Hierarchical, DBSCAN, Gaussian Mixtures

## K-Means
**Objective (inertia):** $J=\sum_k\sum_{x\in C_k}\|x-\mu_k\|^2$. **Lloyd's algorithm** alternates two steps, each of which can only *decrease* $J$:
1. *Assign* every point to its nearest centroid (optimal for fixed centroids);
2. *Update* each centroid to the mean of its members (the mean minimises squared distance to a set of points).

$J$ is bounded below, so the process converges - but only to a *local* minimum. Two mitigations are built in:
* **K-Means++ seeding:** the first centre is uniform; each next centre is drawn with probability proportional to $D(x)^2$, its squared distance to the nearest chosen centre (Arthur & Vassilvitskii: $O(\log k)$-competitive in expectation).
* **`n_init` restarts**, keeping the lowest inertia. Empty clusters are re-seeded at the worst-fit point.

`model.history` stores the centroids after every iteration - `animate_kmeans` turns it into a GIF (`docs/images/kmeans.gif`).

**Choosing $k$:** `elbow_curve` (inertia vs $k$; look for the bend) and `silhouette_curve` ($s=\frac{b-a}{\max(a,b)}$, with $a$ the mean intra-cluster distance and $b$ the mean distance to the nearest other cluster; maximise it).

## Agglomerative (hierarchical) clustering
Start with $n$ singleton clusters and repeatedly merge the closest pair. Cluster distances update by the Lance-Williams formula:
single $\min$, complete $\max$, average (size-weighted mean), **Ward** (minimise the increase in within-cluster variance):
$$d(ij,m)=\tfrac{(n_i+n_m)d(i,m)+(n_j+n_m)d(j,m)-n_m\,d(i,j)}{n_i+n_j+n_m}\quad(\text{on squared distances}).$$
`linkage()` returns a SciPy-compatible matrix (cached row minima make it ~$O(n^2)$); `fcluster` cuts it; `plot_dendrogram` draws it.

## DBSCAN
A point is a *core* point if $\ge$ `min_samples` points lie within `eps`. Clusters are the connected components of core points (plus their border points); everything else is noise (`-1`).
It finds arbitrarily shaped clusters (two moons!) and needs no $k$, but cannot cope with widely varying density.

## Gaussian mixture model (EM)
Model the density as $p(x)=\sum_k\pi_k\mathcal N(x;\mu_k,\Sigma_k)$.
* **E-step:** responsibilities $r_{ik}=\pi_k\mathcal N(x_i;\mu_k,\Sigma_k)/p(x_i)$ (computed in log space with Cholesky factors).
* **M-step:** $N_k=\sum_ir_{ik}$, $\pi_k=N_k/n$, $\mu_k=\tfrac1{N_k}\sum_ir_{ik}x_i$, $\Sigma_k=\tfrac1{N_k}\sum_ir_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top+\epsilon I$.

EM never decreases the log-likelihood (asserted in the tests). Initialised from K-Means. `bic` / `aic` select the number of components; `sample` draws new data. K-Means is the limit of GMM with spherical, equal, vanishing covariances.

## Complexity
K-Means $O(Tnkd)$. Agglomerative $O(n^2)$ time and memory. DBSCAN $O(n^2)$ (dense distance matrix). GMM $O(Tnkd^2)$.

## Example
```python
from mlforge.cluster import KMeans, AgglomerativeClustering, DBSCAN, GaussianMixture
km = KMeans(4, n_init=10, random_state=0).fit(X)
km.cluster_centers_, km.inertia_, km.labels_
```

## Verification
K-Means inertia equals scikit-learn's; all four linkages give ARI $=1.0$ against scikit-learn and heights equal SciPy's to $10^{-15}$;
DBSCAN labels are identical; GMM log-likelihood and BIC agree to $10^{-4}$.

## References
Lloyd (1982). Arthur & Vassilvitskii (2007) *k-means++.* Ester et al. (1996) *DBSCAN.* Dempster, Laird & Rubin (1977) *EM.* Rousseeuw (1987) *Silhouettes.*
