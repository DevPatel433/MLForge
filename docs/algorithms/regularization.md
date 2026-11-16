# Ridge, Lasso & Elastic Net

## Intuition
Penalising large coefficients reduces variance (overfitting) at the price of a little bias. **L2** (Ridge) shrinks all
coefficients smoothly; **L1** (Lasso) drives some *exactly* to zero (feature selection); **Elastic Net** mixes both and copes with correlated features.

## Objectives (scikit-learn conventions, so results are directly comparable)
$$\text{Ridge: } \|y-Xw\|^2+\alpha\|w\|_2^2 \qquad
\text{Lasso: } \tfrac1{2n}\|y-Xw\|^2+\alpha\|w\|_1 $$
$$\text{ElasticNet: } \tfrac1{2n}\|y-Xw\|^2+\alpha\rho\|w\|_1+\tfrac{\alpha(1-\rho)}2\|w\|_2^2 .$$

## Derivations
**Ridge** is differentiable: $\nabla = -2X^\top(y-Xw)+2\alpha w=0\Rightarrow w=(X^\top X+\alpha I)^{-1}X^\top y$. The $\alpha I$ term also makes the
system always invertible. MLForge solves it with `numpy.linalg.solve` on centred data.

**Lasso / Elastic Net** are not differentiable at $0$. Two solvers are implemented:

*Coordinate descent* minimises over one coordinate at a time. With $\rho_j=\tfrac1n x_j^\top(r+x_jw_j)$ ($r$ the current residual) and $z_j=\tfrac1n\|x_j\|^2$:

$$w_j \leftarrow \frac{S(\rho_j,\ \alpha\rho)}{z_j+\alpha(1-\rho)},\qquad S(a,\lambda)=\operatorname{sign}(a)\max(|a|-\lambda,0)$$

where $S$ is the **soft-threshold** operator - the source of exact zeros. The residual is updated incrementally so a full sweep costs $O(nd)$.

*ISTA* (proximal gradient): a gradient step on the smooth part, then soft-threshold, with step $1/L$, $L=\sigma_{\max}(X)^2/n+\alpha(1-\rho)$.

## Complexity
Ridge: $O(nd^2+d^3)$. Coordinate descent: $O(nd)$ per sweep, number of sweeps depends on conditioning and `tol`.

## Example
```python
from mlforge.linear_model import Ridge, Lasso, ElasticNet
Lasso(alpha=0.1).fit(X, y).coef_            # exact zeros for irrelevant features
from mlforge.visualization import plot_regularization_path   # coefficient paths as alpha grows
```

## Verification
Ridge, Lasso and ElasticNet match scikit-learn to $\sim10^{-15}$ in the benchmark (the parity tests assert $10^{-10}$ / $10^{-6}$), CD and ISTA agree with each other, and a
sufficiently large $\alpha$ zeroes every Lasso coefficient (all in the test-suite).

## References
Tibshirani (1996) *Regression shrinkage and selection via the lasso.* Zou & Hastie (2005) *Regularization and variable selection via the elastic net.* Friedman, Hastie & Tibshirani (2010) *Regularization paths for GLMs via coordinate descent.*
