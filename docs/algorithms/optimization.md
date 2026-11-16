# Optimization

All gradient-based models in MLForge (linear / logistic / softmax regression, the linear SVM, the MLP)
share one training loop, `mlforge.optimization.minimize_batches`, and any of the optimizers below can be
plugged in by name (`optimizer="adam"`), class or instance.

## Intuition

Training means minimising a loss $L(\theta)$. The gradient $g = \nabla L(\theta)$ points uphill, so we step
the other way. The optimizers differ in *how* they turn the raw gradient into a step.

## Update rules

| Optimizer | Update ($g_t$ = gradient at step $t$, $\eta$ = learning rate) |
|---|---|
| Gradient descent | $\theta \leftarrow \theta - \eta g_t$ |
| Momentum | $v \leftarrow \beta v + \eta g_t,\quad \theta \leftarrow \theta - v$ |
| Nesterov | $v \leftarrow \beta v + \eta g_t,\quad \theta \leftarrow \theta - (\beta v + \eta g_t)$ |
| AdaGrad | $G \leftarrow G + g_t^2,\quad \theta \leftarrow \theta - \eta\, g_t / (\sqrt{G}+\epsilon)$ |
| RMSProp | $s \leftarrow \beta s + (1-\beta) g_t^2,\quad \theta \leftarrow \theta - \eta\, g_t/(\sqrt{s}+\epsilon)$ |
| Adam | $m \leftarrow \beta_1 m + (1-\beta_1) g,\; v \leftarrow \beta_2 v + (1-\beta_2) g^2,\; \theta \leftarrow \theta - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ with $\hat m = m/(1-\beta_1^t)$, $\hat v = v/(1-\beta_2^t)$ |

**Batch / mini-batch / stochastic** gradient descent are the same update; they differ in how many samples
produce $g_t$: all $n$ (batch), `batch_size` (mini-batch) or one (`batch_size=1`, SGD). Mini-batches
trade gradient noise for many more updates per pass over the data.

### Why momentum helps
On a long narrow valley (an ill-conditioned quadratic $\tfrac12 \theta^\top A \theta$) plain GD must use a
step small enough for the steepest direction, so it crawls along the shallow one. Momentum accumulates
the consistent component of the gradient (along the valley) while oscillating components cancel.
Run `examples/make_gallery.py optimizer_race` to watch six optimizers cross the Rosenbrock valley.

### Adam's bias correction
Initialising $m_0=v_0=0$ biases the moving averages towards zero early on; dividing by $1-\beta^t$ removes
that bias. A consequence, checked by a unit test: Adam's very first step has magnitude $\approx \eta$ per coordinate regardless
of the gradient's scale.

## Learning-rate schedules
`StepDecay`, `ExponentialDecay`, `InverseTimeDecay`, `CosineAnnealing`, `WarmupSchedule`: callables
$t \mapsto$ multiplier, passed as `Optimizer(..., schedule=...)`. The step index $t$ is zero-based.

## Complexity
Every step is $O(d)$ time and memory ($O(2d)$ for Adam) on top of the cost of computing the gradient.

## Example
```python
from mlforge.optimization import minimize, rosenbrock, rosenbrock_grad
path = minimize(rosenbrock, rosenbrock_grad, [-1.5, 2.0], optimizer="adam", n_iters=3000, learning_rate=0.05)
print(path[-1])        # ~[1, 1]
```

## Verification
`tests/test_optimization.py`: every optimizer minimises a quadratic bowl; Adam / momentum solve Rosenbrock; analytic gradients of the
test surfaces are checked numerically; schedules are checked value-by-value (this caught an off-by-one bug in v0.2 development).

## References
- Ruder (2016). *An overview of gradient descent optimization algorithms.* arXiv:1609.04747
- Kingma & Ba (2015). *Adam: A method for stochastic optimization.*
- Sutskever et al. (2013). *On the importance of initialization and momentum in deep learning.*
