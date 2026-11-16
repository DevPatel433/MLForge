# Logistic, Softmax Regression & Perceptron

## Intuition
Pass a linear score through a squashing function to obtain class probabilities and fit by maximum likelihood.

## Derivation (binary)
$p=\sigma(z)$, $z=w^\top x+b$. The negative log-likelihood (cross-entropy) is

$$L=-\tfrac1n\sum_i\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big]+\tfrac1{2Cn}\|w\|^2 .$$

Using $\sigma'(z)=\sigma(1-\sigma)$ everything cancels to the strikingly simple gradient
$$\nabla_wL=\tfrac1nX^\top(p-y)+\tfrac{w}{Cn},\qquad \partial_bL=\tfrac1n\sum_i(p_i-y_i).$$
(`C` is the inverse regularisation strength, as in scikit-learn.) The loss is computed as $\log(1+e^{z})-yz$ with `logaddexp` for numerical stability.

**Newton / IRLS** (`solver="newton"`): the Hessian is $H=\tfrac1nX^\top\operatorname{diag}(p(1-p))X+\text{reg}$, and
$\theta\leftarrow\theta-H^{-1}\nabla L$ converges in a handful of iterations - quadratically near the optimum.

**Softmax regression** generalises this: $P_{ik}=\operatorname{softmax}(XW+b)_{ik}$, loss $-\tfrac1n\sum_{i}\log P_{i,y_i}$ and
$\nabla_W=\tfrac1nX^\top(P-Y)$ with $Y$ the one-hot matrix.

## Perceptron
Rosenblatt's rule: on a mistake ($t_i f(x_i)\le0$, $t_i\in\{\pm1\}$) update $w\leftarrow w+\eta t_ix_i$. By the perceptron convergence theorem it makes a finite number of mistakes iff the data are linearly separable
(`errors_` records mistakes per epoch). Multiclass uses one-vs-rest.

## Caveat: separable data
Without regularisation, the MLE on perfectly separable data does not exist (weights grow without bound). Use `C` or early stopping.
The parity tests use overlapping classes for exactly this reason.

## Complexity
GD: $O(Tnd)$. Newton: $O(nd^2+d^3)$ per iteration. Softmax multiplies by the number of classes $k$.

## Example
```python
from mlforge.linear_model import LogisticRegression, SoftmaxRegression
clf = LogisticRegression(solver="newton", C=1.0).fit(X, y)
clf.predict_proba(X)[:, 1]
```

## Verification
All three gradients pass finite-difference checks; Newton matches scikit-learn's L-BFGS solution to $10^{-5}$ (probabilities $10^{-6}$).

## References
Bishop, *PRML*, §4.3. Rosenblatt (1958) *The perceptron.* Murphy, *Machine Learning: A Probabilistic Perspective*, ch. 8.
