# Automatic Differentiation (`mlforge.autograd`)

A ~200-line reverse-mode autodiff engine on NumPy: a `Tensor` records the operations that produced it, and `.backward()` replays the graph in reverse applying the chain rule.

```python
from mlforge.autograd import Tensor
x = Tensor([2.0, 3.0], requires_grad=True)
y = (x * x).sum() + x.exp().sum()
y.backward()
x.grad                     # 2x + e^x  ->  [11.389, 26.086]
```

## How it works
* Each operation creates an output `Tensor` that stores its parents and a small closure computing `d(parents)` from `d(output)`.
* `backward()` topologically sorts the graph and calls the closures from the output backwards, **accumulating** (`+=`) into `grad`, so a tensor used several times gets the sum of its contributions.
* **Broadcasting:** when an operand was broadcast in the forward pass its gradient is summed back over the broadcast axes (`_unbroadcast`).
* **Matmul** handles matrix-matrix, matrix-vector, vector-matrix and vector-vector cases: $\partial/\partial A=G B^\top$, $\partial/\partial B=A^\top G$.
* `no_grad()` disables graph building for inference.

Supported: `+ - * / **`, `@`, `sum`, `mean` (with axes / keepdims), `exp`, `log`, `relu`, `tanh`, `sigmoid`, `log_softmax`, `reshape`, `.T`.

## Verification
Every operation is checked against finite differences (`tests/test_autograd.py`), including broadcasting and all four matmul shapes, and a 2-layer network is trained on XOR using nothing but `Tensor` operations.

## Why include it?
It makes the connection between *hand-derived* gradients (used by the library's models) and *mechanically derived* ones concrete: the two can be compared on the same model.

## References
Baydin et al. (2018) *Automatic differentiation in machine learning: a survey.* Karpathy's *micrograd* (the inspiration).
