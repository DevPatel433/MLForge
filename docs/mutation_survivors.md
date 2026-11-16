# Mutation survivors: review notes

Final sampled run: **207 of 220 mutants killed (94.1%)**; the 13 survivors below were each reviewed by hand. Earlier in development the same procedure
found 30+ genuine test gaps (all closed with tests); every survivor left is *equivalent in outcome* - it cannot change what the function returns.
Regenerate the list with `python tools/mutation_test.py --survivors`.

| Module | Mutation | Why it cannot change behaviour |
|---|---|---|
| `utils/math_utils.py` | `np.clip(z, -500, 500)`: bound 500 to 501 (sigmoid) | `exp(-500)` and `exp(-501)` are both indistinguishable from 0 next to 1 in float64 |
| `neural_network/recurrent.py` | the same clip bound in the LSTM's sigmoid | same reason |
| `utils/pairwise.py` (3 mutants) | chunk-size arithmetic: `max(1, ...)` constants and `*` to `/` | chunking only bounds memory; the result is identical for any chunk size (tested with forced tiny chunks) |
| `decomposition/pca.py` | `signs[signs == 0] = 1`: constant 1 to 2 | unreachable: SVD rows are orthonormal, so no component has an all-zero largest entry |
| `linear_model/regularized.py` | ISTA stopping test `<` to `<=` | differs only when the change equals the tolerance exactly |
| `linear_model/regularized.py` | Lipschitz constant `+ l2` to `- l2` | a slightly larger ISTA step still converges to the same optimum; only the iteration count changes |
| `neural_network/conv.py` (2 mutants) | `reshape(F, -1)` to `reshape(F, -2)` | NumPy treats any negative dimension as "infer this one" |
| `autograd/tensor.py` | `a.ndim > 1` threshold in the matmul backward pass | the extra leading axis is summed away by `_unbroadcast`, giving identical gradients |
| `autograd/tensor.py` | `ga.reshape(...) if a.ndim == 1 else ga` (`==` to `!=`) | the same `_unbroadcast` normalisation makes both branches equal |
| `tree/hist_boosting.py` | leaf marker `-1` to `-2` for `right` | only `left == -1` is ever consulted to detect leaves |
