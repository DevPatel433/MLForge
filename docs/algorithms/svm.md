# Support Vector Machines

## Intuition
Among all separating hyperplanes choose the one with the **widest margin**. Points that touch the margin - the *support vectors* - alone determine it.

## Primal: `LinearSVC`
With labels $t_i\in\{\pm1\}$ and the soft-margin objective
$$\min_{w,b}\ \tfrac12\|w\|^2+C\sum_i\max(0,1-t_i(w^\top x_i+b)).$$
`LinearSVC` minimises its mean form $\tfrac1n\sum_i\text{hinge}_i+\tfrac1{2Cn}\|w\|^2$ by (sub)gradient descent: the hinge's subgradient is $-t_ix_i$ for margin-violating points and $0$ otherwise.
`loss="squared_hinge"` is smooth. Large $C$ = narrow margin / few violations; small $C$ = wide margin. Multiclass is one-vs-rest.

## Dual and kernels: `SVC`
The Lagrangian dual is
$$\max_\alpha\ \sum_i\alpha_i-\tfrac12\sum_{ij}\alpha_i\alpha_jt_it_jK(x_i,x_j)\quad\text{s.t. }0\le\alpha_i\le C,\ \sum_i\alpha_it_i=0,$$
with decision function $f(x)=\sum_i\alpha_it_iK(x_i,x)+b$. Only support vectors have $\alpha_i>0$. Replacing $x^\top x'$ by a kernel gives a non-linear boundary without ever computing the feature map:

| kernel | $K(x,x')$ |
|---|---|
| linear | $x^\top x'$ |
| poly | $(\gamma x^\top x'+c_0)^d$ |
| rbf | $\exp(-\gamma\|x-x'\|^2)$ |
| sigmoid | $\tanh(\gamma x^\top x'+c_0)$ |

### SMO
Sequential Minimal Optimisation optimises the dual two multipliers at a time (the smallest subproblem that respects $\sum\alpha_it_i=0$), which has a closed-form solution:
$$\alpha_j^{new}=\text{clip}\Big(\alpha_j+\tfrac{t_j(E_i-E_j)}{\eta},L,H\Big),\ \ \eta=K_{ii}+K_{jj}-2K_{ij},\ \ \alpha_i^{new}=\alpha_i+t_it_j(\alpha_j-\alpha_j^{new}),$$
$E=f-t$ the error, $[L,H]$ the box bounds. MLForge uses Platt's second-choice heuristic (maximise $|E_i-E_j|$ among non-bound points) with an error cache. Multiclass uses one-vs-one voting.

## Complexity
Kernel matrix $O(n^2)$ memory; SMO is roughly $O(n^2)$-$O(n^3)$ time. Prediction $O(n_{SV}d)$.

## Example
```python
from mlforge.svm import SVC
m = SVC(C=1.0, kernel="rbf", gamma="scale").fit(X, y)
m.support_vectors_, m.dual_coef_, m.decision_function(X)
```

## Verification
The dual constraints ($0\le\alpha\le C$, $\sum\alpha t=0$) are asserted on SMO's output; RBF SVMs reach the same predictions as scikit-learn (>= 97% agreement) with essentially the same number of support vectors (e.g. 44 vs 44 on two-moons); the hinge-loss gradient is checked numerically.

## References
Cortes & Vapnik (1995). Platt (1998) *Sequential minimal optimization.* Schölkopf & Smola, *Learning with Kernels.*
