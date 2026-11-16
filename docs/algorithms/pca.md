# Principal Component Analysis

## Intuition
Find the orthogonal directions along which the data vary the most and describe each sample by its coordinates along the top few of them.

## Derivation
Centre the data and form the covariance $\Sigma=\tfrac1{n-1}X_c^\top X_c$. Maximising the projected variance $w^\top\Sigma w$ subject to $\|w\|=1$ is a Rayleigh quotient problem: the optimum is the top **eigenvector** of $\Sigma$, with variance equal to its **eigenvalue** $\lambda_1$; the next components are the following eigenvectors (orthogonal because $\Sigma$ is symmetric).

Equivalently, with the thin SVD $X_c=U S V^\top$: the principal axes are the rows of $V^\top$ and $\lambda_i=s_i^2/(n-1)$. The SVD route (`solver="svd"`, default) avoids squaring the condition number of $X_c$; `solver="eigh"` is the textbook covariance route.

* **Explained variance ratio** $\lambda_i/\sum_j\lambda_j$; `n_components=0.95` keeps the fewest components reaching 95%.
* **Whitening** divides each score by $\sqrt{\lambda_i}$ (unit variance, decorrelated).
* **Reconstruction** $\hat X=ZV_k^\top+\mu$; `reconstruction_error` is the mean squared residual - it equals the sum of the discarded eigenvalues (divided by $d$).
* Signs of eigenvectors are arbitrary; MLForge fixes them (largest-magnitude loading positive) so results are reproducible.

## Complexity
SVD $O(\min(nd^2,n^2d))$; transform $O(ndk)$.

## Example
```python
from mlforge.decomposition import PCA
pca = PCA(n_components=0.95).fit(X_scaled)
Z = pca.transform(X_scaled); pca.explained_variance_ratio_.cumsum()
```

## Verification
Variances equal scikit-learn's to $10^{-9}$ relative, scores agree up to sign to $10^{-8}$; components are orthonormal and scores decorrelated (tests).

## References
Pearson (1901); Hotelling (1933); Jolliffe, *Principal Component Analysis*.
