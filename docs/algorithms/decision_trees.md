# Decision Trees (CART)

## Intuition
Recursively ask the single yes/no question about one feature that makes the two resulting groups as *pure* as possible.

## Impurity and information gain
For class proportions $p_k$ at a node:
$$\text{Gini}=1-\sum_kp_k^2,\qquad \text{Entropy}=-\sum_kp_k\log_2p_k,\qquad \text{Regression: variance } \tfrac1n\sum(y_i-\bar y)^2 .$$
A split into $L,R$ is scored by its **gain** $=I(\text{parent})-\tfrac{n_L}{n}I(L)-\tfrac{n_R}{n}I(R)$ (information gain when $I$ is entropy). The greedy algorithm picks the split with the largest gain and recurses until a stopping rule fires (`max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`, or a pure node).

Zero-gain splits are *accepted* (as in scikit-learn): XOR has no first split that reduces impurity, but depth-2 trees can still solve it.

## Fast split search
For each feature the samples are sorted once; cumulative class counts (or cumulative sums of $y,y^2$ for regression) give the impurity of **every** threshold at once, so a node costs $O(d\,n\log n)$ instead of $O(d\,n^2)$. MLForge vectorises this over *all* features simultaneously with `argsort(axis=0)` and `take_along_axis`.

## Cost-complexity pruning
Grow a large tree $T_0$, then for each internal node $t$ compute the *effective alpha* of collapsing its subtree $T_t$:
$$\alpha_{\text{eff}}(t)=\frac{R(t)-R(T_t)}{|T_t|-1},\qquad R=\text{weighted impurity of leaves}.$$
Repeatedly collapse the weakest link (smallest $\alpha_{\text{eff}}$) while it is $\le$ `ccp_alpha`. `cost_complexity_pruning_path` returns the whole alpha sequence for cross-validation.

## Inspection
`feature_importances_` (total impurity decrease per feature, normalised), `export_text()`, `get_depth()`, `get_n_leaves()`, and `mlforge.visualization.plot_tree` to draw it.

## Complexity
Training $O(d\,n\log n)$ per tree level (balanced trees), prediction $O(\text{depth})$ per sample, memory $O(\text{nodes})$.

## Example
```python
from mlforge.tree import DecisionTreeClassifier
t = DecisionTreeClassifier(max_depth=4, criterion="entropy", ccp_alpha=0.005).fit(X, y)
print(t.export_text(feature_names))
```

## Verification
Number of leaves and depth are identical to scikit-learn on iris / wine / breast-cancer for 12 hyper-parameter settings including pruning; regressor predictions match to $10^{-8}$.

## References
Breiman, Friedman, Olshen & Stone (1984) *Classification and Regression Trees.* Quinlan (1986) *Induction of decision trees.*
