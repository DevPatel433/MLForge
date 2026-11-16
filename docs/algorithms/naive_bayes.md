# Naive Bayes

## Intuition
Apply Bayes' theorem and *pretend* the features are independent given the class:
$$P(c\mid x)\ \propto\ P(c)\prod_jP(x_j\mid c).$$
The assumption is almost always false, yet the classifier is often excellent because only the *ranking* of classes matters.

## Variants
| Class | Feature model $P(x_j\mid c)$ | Fitted parameters |
|---|---|---|
| `GaussianNB` | $\mathcal N(\mu_{cj},\sigma^2_{cj})$ | class means, variances (+ `var_smoothing`) |
| `MultinomialNB` | multinomial over counts | $\theta_{cj}=\dfrac{N_{cj}+\alpha}{N_c+\alpha d}$ (Laplace / Lidstone smoothing) |
| `BernoulliNB` | Bernoulli (feature present / absent) | $\theta_{cj}=\dfrac{N_{cj}+\alpha}{n_c+2\alpha}$, absent features count too |

All computation is in **log space** (sums of log-probabilities, normalised with log-sum-exp), avoiding underflow with many features.
Smoothing prevents a single unseen feature from zeroing a whole class.

### Worked example (tested by hand)
One binary feature, 4 samples per class with $x=1$ in 1/4 of class 0 and 3/4 of class 1. With $\alpha=1$: $P(x{=}1|c_0)=2/6$, $P(x{=}1|c_1)=4/6$, equal priors, so $P(c_1\mid x{=}1)=\frac{4/6}{4/6+2/6}=\frac23$.

## Complexity
Fit $O(nd)$, predict $O(kd)$ per sample - one pass over the data and no iteration.

## Example
```python
from mlforge.bayes import GaussianNB
GaussianNB().fit(X, y).predict_proba(X_new)
```

## Verification
Posterior probabilities match scikit-learn to $10^{-9}$ for all three variants.

## References
Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, ch. 13. Murphy, *Machine Learning: A Probabilistic Perspective*, §3.5.
