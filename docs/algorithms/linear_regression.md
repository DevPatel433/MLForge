# Linear & Polynomial Regression

## Intuition
Model the target as an affine function of the features, $\hat y = w^\top x + b$, and choose $(w,b)$ to minimise the
mean squared error.

## Mathematical derivation
With design matrix $X\in\mathbb R^{n\times d}$:

$$L(w,b)=\frac1n\sum_i(w^\top x_i+b-y_i)^2,\qquad
\nabla_w L=\frac2n X^\top e,\quad \partial_b L=\frac2n\sum_i e_i,\quad e = Xw+b\mathbf 1-y .$$

**Closed form.** Centre $X$ and $y$ (this absorbs the intercept) and set $\nabla_w L=0$ to get the normal equations
$X_c^\top X_c\,w = X_c^\top y_c$. MLForge solves them with `numpy.linalg.lstsq`, which uses an SVD and stays stable when
$X^\top X$ is singular (it returns the minimum-norm solution) - explicitly *not* by forming $(X^\top X)^{-1}$.

**Gradient descent** repeats $w\leftarrow w-\eta\nabla_wL$. $L$ is a convex quadratic with Hessian $\tfrac2n X^\top X$, so GD converges for
$\eta<2/\lambda_{\max}$; unscaled features make this very restrictive, which is why `StandardScaler` matters
(the library's own divergence bug during development was exactly this). Use `solver="normal"` (default) or scale your data.

**Polynomial regression** is linear regression on expanded features
$[x,\,x^2,\dots,x^k]$ (plus cross terms with several inputs). `PolynomialRegression` standardises the expanded
features internally.

## Algorithm
1. (`solver="normal"`) center, `lstsq`, recover $b=\bar y-\bar x^\top w$.
2. (`solver="gd"`) repeat: forward pass, error, gradient, optimizer step; record the loss in `cost_history_`.

## Complexity
Closed form: $O(nd^2)$ time, $O(d^2)$ memory. GD: $O(T\,nd)$ for $T$ epochs; prediction $O(nd)$.

## Example
```python
from mlforge.linear_model import LinearRegression, PolynomialRegression
m = LinearRegression(solver="gd", optimizer="adam", learning_rate=0.1, n_iters=500).fit(X, y)
m.coef_, m.intercept_, m.cost_history_          # fitted parameters and loss per epoch
PolynomialRegression(degree=3).fit(x, y)         # 1-D input is fine
```

## Verification
GD reaches the closed-form solution to $10^{-5}$; gradients pass finite-difference checks (`tests/test_gradients.py`);
OLS matches scikit-learn to $10^{-10}$ (`tests/test_sklearn_parity.py`).

## References
Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, ch. 3. Bishop, *PRML*, ch. 3.
