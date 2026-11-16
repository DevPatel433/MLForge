# Random Forests & Gradient Boosting

## Random forest
**Intuition.** Single trees have low bias but high variance. Averaging many *decorrelated* trees keeps the bias and slashes the variance.

**Algorithm.** For each of `n_estimators` trees: draw a bootstrap sample of the $n$ rows; grow a tree where each split only considers a random subset of
`max_features` features (default $\sqrt d$ for classification); average the class probabilities (soft voting) or predictions.

**Out-of-bag estimate.** Each bootstrap leaves out $\approx(1-1/n)^n\to e^{-1}\approx 36.8\%$ of rows. Predicting every row with only the trees that never saw it gives a free generalisation estimate (`oob_score=True`).

If the trees' individual variance is $\sigma^2$ and their pairwise correlation $\rho$, the ensemble variance is $\rho\sigma^2+\tfrac{1-\rho}{B}\sigma^2$ - feature subsampling lowers $\rho$.

## Gradient boosting
**Intuition.** Build the model stagewise: each new small tree fits what the current ensemble gets wrong.

Treat the model $F$ as a parameter and do gradient descent in function space. At stage $m$ fit a regression tree $h_m$ to the negative gradient
$r_i=-\partial L(y_i,F(x_i))/\partial F(x_i)$, then $F\leftarrow F+\eta\,h_m$.

* **Squared loss:** $r_i=y_i-F(x_i)$ - the ordinary residuals.
* **Binary deviance:** $r_i=y_i-p_i$ with $p=\sigma(F)$. Tree leaves are refined with one Newton step $\gamma_\ell=\sum_{i\in\ell}r_i/\sum_{i\in\ell}p_i(1-p_i)$.
* **Multiclass:** one tree per class per stage on softmax residuals, with Friedman's $(K-1)/K$ leaf scaling.

`learning_rate` (shrinkage), `max_depth` (weak learners), `subsample` (stochastic boosting) and the number of stages trade off against each other; `staged_predict` / `train_loss_` show how the fit evolves.

## Histogram-based gradient boosting (`HistGradientBoosting*`)
Exact-split boosting re-sorts feature values at every node. The histogram variant (LightGBM / scikit-learn's `HistGradientBoosting`) works on **binned** features:

1. **Binning.** Each feature is discretised once into at most 256 ordered bins (midpoints between distinct values, or quantiles); bin ids are `uint8`.
2. **Histograms.** For a node, the per-(feature, bin) sums of gradients $g$, hessians $h$ and counts come from a single `np.bincount`. The best split of *every* feature follows from cumulative sums.
3. **Subtraction trick.** A child's histogram is the parent's minus its sibling's, so only the *smaller* child is ever scanned.
4. **Leaf-wise growth.** The leaf with the largest gain is split next, up to `max_leaf_nodes`. With Newton boosting the split gain is
   $\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{G^2}{H+\lambda}$ and a leaf's value is $-G/(H+\lambda)$ ($\lambda$ = `l2_regularization`).

Per-node cost is $O(\text{bins}\times\text{features})$ for the split search plus $O(n_{\text{smaller child}}\times\text{features})$ for the histogram, so training time grows almost flat in $n$:
**2.3x faster than exact boosting at 2,000 rows, 6.3x at 8,000, 12.6x at 32,000** (capacity-matched, `benchmarks/scaling_hist_boosting.py`).
On the small 1,400-row benchmark it is *slower* than scikit-learn (4.5x regression, 1.5x multiclass) at the same quality.

**Verification:** the histogram split search is compared with an exhaustive search over every (feature, bin) pair; the subtraction trick is checked to be exact; binning is checked to be monotone and to separate values one floating-point step apart; held-out quality matches scikit-learn's within 0.03.

## Complexity
Forest: $B\times$ tree cost (embarrassingly parallel, but run sequentially here). Boosting: $M$ sequential trees.

## Example
```python
from mlforge.tree import RandomForestClassifier, GradientBoostingRegressor
rf = RandomForestClassifier(200, oob_score=True, random_state=0).fit(X, y); rf.oob_score_
gb = GradientBoostingRegressor(300, learning_rate=0.05, max_depth=3, subsample=0.8).fit(X, y)
```

## Verification
Regression boosting reproduces scikit-learn's $R^2$ to $10^{-4}$ on the benchmark data; forests / boosting stay within 3 accuracy points of scikit-learn on breast-cancer (randomised algorithms cannot be compared bit-for-bit).

## References
Breiman (2001) *Random forests.* Friedman (2001) *Greedy function approximation: a gradient boosting machine.*
