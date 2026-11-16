# How do you know your machine-learning implementation is correct?

*Lessons from building MLForge, a pure-NumPy machine-learning library, with an evidence-first workflow.*

"It trains and the loss goes down" is the usual evidence offered for a from-scratch implementation. It is weak evidence.
A gradient that is wrong by a constant factor still decreases the loss (just at the wrong speed). A function that returns the
F1 score where it should return recall still produces plausible numbers between 0 and 1. A tree that cannot split two
neighbouring floating-point values will pass every test you wrote with sensible-looking data. This essay describes the layers of
evidence MLForge uses instead, and - more usefully - the *specific bugs each layer caught* that the previous layers missed.

## 1. Numerical gradient checks: is the derivative right?

Any model trained by gradient descent has a hand-derived gradient, and hand-derived gradients are easy to get subtly wrong.
The remedy is old and cheap: compare the analytic gradient with a central finite difference,

$$\frac{L(\theta+\epsilon e_i)-L(\theta-\epsilon e_i)}{2\epsilon}\approx\frac{\partial L}{\partial\theta_i},$$

for every parameter. MLForge exposes this as `mlforge.utils.check_gradient` and applies it to every loss, to whole multi-layer
networks, to convolutions (stride and padding variants), max-pooling, and back-propagation through time in an LSTM. The dashboard
([correctness](correctness.md)) re-computes all of them on every run; relative errors are around $10^{-10}$ against a pass
threshold of $10^{-5}$. The test-suite also checks that a *deliberately wrong* gradient is detected - a checker that cannot fail proves nothing.

What it cannot do: tell you the **loss** is the right loss, or that the optimizer is correct. For those you need the next layers.

## 2. Reference parity: does it agree with an established implementation?

Where the mathematics is deterministic, two correct implementations must agree. MLForge compares against scikit-learn and SciPy
at explicit tolerances (OLS and Ridge to $10^{-10}$, PCA variances to $10^{-9}$, k-NN and Naive Bayes probabilities to $10^{-9}$,
hierarchical-clustering heights to $10^{-10}$). Two lessons were less obvious than the tolerances:

* **Parity can be ill-posed.** Unregularised logistic regression on perfectly separable data has no finite optimum - the weights diverge - so two correct
  solvers disagree by construction. The test must use overlapping classes, where a maximum-likelihood solution exists.
* **Parity can be undefined.** On data with tied distances (the iris dataset has duplicate points) *which* of several equidistant neighbours is returned is
  implementation-defined. The right test compares what *is* defined - the neighbour **distances** - not the indices. Likewise, random forests match
  scikit-learn statistically, never bit-for-bit.

## 3. Properties: does it hold for *every* input, not just the ones I thought of?

Property-based testing (Hypothesis) states a law and searches for counterexamples. MLForge has 21 such laws: scalers round-trip, folds partition the data,
softmax rows sum to one and are shift-invariant, pairwise distances satisfy the triangle inequality, OLS residuals are orthogonal to the features, an unrestricted
tree memorises any labelling of distinct points... Two genuine bugs fell out within minutes of writing them:

1. **Constant column, garbage output.** For a column of identical floats like `114.7758...`, rounding in the mean makes `std()` come out as $\approx10^{-14}$ instead of 0.
   The scaler's `scale == 0` guard did not fire, so it divided rounding noise by itself and returned $\pm1$ instead of 0. Hand-written tests
   had used "nice" constants like `5.0`, whose mean is exact. The fix compares against a tolerance relative to machine precision.
2. **Infinite recursion.** For two feature values exactly one floating-point step apart, the midpoint `(a+b)/2` rounds *up to b*, so `x <= threshold` sends both points left,
   the child equals the parent, and tree growth recurses until the stack overflows. scikit-learn guards against exactly this; MLForge did not.

Both are now deterministic regression tests, and the second test was checked to *fail* when the fix is removed.

## 4. Mutation testing: would my tests notice if the code were wrong?

Coverage tells you which lines ran, not whether any assertion would fail if they were wrong. Mutation testing makes the question concrete: inject one small fault at a time
(`+`→`-`, `<`→`<=`, `and`→`or`, a constant off by one), run the tests, and see whether any fails. MLForge's small dependency-free tool
(`tools/mutation_test.py`) samples mutants per module; the score is the fraction killed. Surviving mutants are triaged by hand into *test gaps* and *equivalent mutants* (changes that cannot alter behaviour).

The most instructive survivors:

* **`recall_score` mutated to return F1 - and all tests passed.** The test data had precision exactly equal to recall, so recall and F1 coincide. The metric was untested in the one way that matters.
* **Only `GradientDescent` had ever been tested with a learning-rate schedule**, and *no test checked RMSProp's actual update formula* - "minimises a quadratic bowl" passes for many wrong formulas.
* **ReLU's backward pass was only tested on positive inputs**, because the gradient-check helper drew values in $[0.5, 1.5]$ to keep `log` safe.
* **Documented default hyper-parameters were never asserted**, so changing `ddof=0` to `1` in a covariance function broke nothing.

Fixing these raised sampled mutation scores module by module (see the dashboard for current numbers). The method has real limits: it samples rather than enumerates, equivalent mutants must be judged by a human, and it only measures the tests you have, not the properties you forgot.

One methodological trap deserves its own sentence: **a mutation run is only valid if the unmutated tests pass.** If the baseline already fails, every mutant looks "killed". The tool refuses to run otherwise - after the first version of it did exactly that.

## 5. End-to-end results you can reproduce

Finally, real data. `examples/mnist_cnn.py` downloads MNIST, **verifies SHA-256 checksums**, trains a 5,258-parameter convolutional network from scratch in NumPy on one CPU core, and writes every epoch's metrics, the exact configuration and the learned weights to JSON / `.npz`.
Re-evaluating the saved weights through a separate code path (`--eval-only`) must reproduce the reported test accuracy exactly - and does.

## What this does not prove

None of this is a proof of correctness. Tests sample behaviour; mutation scores are estimates; parity with scikit-learn shows agreement, not truth (the reference could be wrong, though that is unlikely for well-trodden algorithms).
The claim is narrower and checkable: **every number in the project's documentation is produced by a script in the repository, and the checks that guard it are themselves measured.**

## A checklist you can reuse

1. Write the gradient check *before* the optimizer. Make sure it can fail.
2. Pick parity tolerances deliberately, and write down where parity is impossible and what invariant replaces it.
3. State the laws your code must obey and let a property-based tester attack them - especially numerical edge cases (constant data, ties, one-ulp neighbours, empty classes).
4. Run mutation testing on the numerical core. Treat every survivor as a question: gap or equivalent?
5. Never trust a claim you typed. Generate documentation numbers from measured artefacts, and re-read your own notebook outputs - one of MLForge's notebooks claimed gradient descent matched the closed form until reading the output showed it had not converged.
