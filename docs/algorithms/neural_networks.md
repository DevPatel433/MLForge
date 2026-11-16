# Neural Networks (MLP)

## Intuition
Stack affine maps and non-linearities: $h_{l}=\phi(h_{l-1}W_l+b_l)$. Non-linearities are what let the network represent functions a linear model cannot - the classic example is **XOR**, which `LogisticRegression` provably cannot fit but a 1-hidden-layer MLP can (both are tests).

## Forward / backward
For layer outputs $z_l=h_{l-1}W_l+b_l$, $h_l=\phi(z_l)$ and loss $L$, backpropagation applies the chain rule from the output down:
$$\delta_L=\frac{\partial L}{\partial z_L},\qquad \frac{\partial L}{\partial W_l}=h_{l-1}^\top\delta_l,\quad \frac{\partial L}{\partial b_l}=\textstyle\sum_{\text{batch}}\delta_l,\quad \delta_{l-1}=(\delta_lW_l^\top)\odot\phi'(z_{l-1}).$$

* **Softmax + cross-entropy** are fused, so $\delta_L=(P-Y)/n$ - numerically stable and exact.
* Activations: ReLU ($\phi'=\mathbb 1[z>0]$), LeakyReLU, Sigmoid ($\sigma(1-\sigma)$), Tanh ($1-\tanh^2$), Softmax layer (full Jacobian-vector product $s\odot(g-\langle g,s\rangle)$).
* **Initialisation** matters: He ($\sqrt{2/n_{in}}$) keeps ReLU activations' variance stable; Xavier ($\sqrt{2/(n_{in}+n_{out})}$) suits tanh / sigmoid.
* **Dropout** (inverted): during training zero each unit with probability $p$ and scale survivors by $1/(1-p)$ so $\mathbb E[h]$ is unchanged and inference needs no rescaling.
* **L2 regularisation** adds $\lambda W$ to $\partial L/\partial W$. **Early stopping** monitors validation loss and restores the best weights.

## API
```python
from mlforge.neural_network import MLPClassifier, Sequential, Dense, ReLU, Dropout, CrossEntropyLoss
clf = MLPClassifier(hidden_sizes=(64, 32), activation="relu", optimizer="adam", dropout=0.1,
                    l2=1e-4, early_stopping=True, n_epochs=200, random_state=0).fit(X, y)
clf.history_            # {'loss', 'metric', 'val_loss', 'val_metric'} per epoch

net = Sequential([Dense(2, 16), ReLU(), Dense(16, 2)])        # build your own
net.fit(X, y, CrossEntropyLoss(), optimizer="adam", epochs=100)
print(net.summary())
```
Any optimizer from `mlforge.optimization` works, each parameter tensor gets its own optimizer state.

## Convolutional layers (`Conv2D`, `MaxPool2D`, `Flatten`)
Images are `(N, C, H, W)`. A convolution layer computes $y_{n,f,i,j}=b_f+\sum_{c,u,v}W_{f,c,u,v}\,x_{n,c,\,is+u,\,js+v}$ (cross-correlation, with stride $s$ and zero padding).
It is implemented with **im2col**: every receptive field is unrolled into one row of a matrix, turning the layer into a single matrix multiplication,

$$\text{out}=\text{cols}\,W_{2D}^\top+b,\qquad dW=g_{2D}^\top\,\text{cols},\qquad d\,\text{cols}=g_{2D}\,W_{2D},\qquad dx=\text{col2im}(d\,\text{cols}),$$

where `col2im` scatter-adds each kernel tap back into the (padded) input gradient. `MaxPool2D` routes the gradient to exactly one input per window (the first maximum on ties).

## Recurrent layer (`LSTM`)
For input $(N,T,D)$ the gates share one affine map, $[i\,f\,g\,o]=x_tW_x+h_{t-1}W_h+b$, with $c_t=\sigma(f)c_{t-1}+\sigma(i)\tanh(g)$ and $h_t=\sigma(o)\tanh(c_t)$.
Backpropagation through time walks the cached forward pass in reverse, carrying $dh$ and $dc$. The forget-gate bias starts at 1 so the cell remembers by default.
An LSTM learns the classic *adding problem* (memory across time steps) well below the predict-the-mean baseline (tested).

## Real data: MNIST
`mlforge.datasets.load_mnist()` downloads the real 60,000 / 10,000 split, **verifies SHA-256 checksums**, and caches it. `examples/mnist_cnn.py` trains from scratch on one CPU core:

| Model | Parameters | Epochs | Test accuracy | Training time |
|---|---:|---:|---:|---:|
| CNN: conv3x3(8) - pool - conv3x3(16) - pool - dense | 5,258 | 5 | **98.57%** | ~97 s |
| MLP: 784 - 128 - 64 - 10 (dropout 0.1) | 109,386 | 8 | 97.82% | ~17 s |

Training is deterministic (two independent runs gave identical accuracy); the log, configuration and weights are stored in `benchmarks/results/`, and `--eval-only`
re-evaluates the saved weights through a separate path. Figures: `examples/mnist_report.py`.

## Verification: gradient checking
Backprop is easy to get subtly wrong. `tests/test_gradients.py` compares the analytic gradient of whole networks to central finite differences
$\frac{L(\theta+\epsilon e_i)-L(\theta-\epsilon e_i)}{2\epsilon}$ for every activation, three losses and with L2 - relative errors are $10^{-9}$-$10^{-11}$. Convolutions (stride 1-3, padding 0-2), pooling, a whole conv-pool-conv-dense network and the LSTM (last state and full sequence) pass the same check, and the convolution forward pass is compared with an independent naive-loop implementation.
On the 8x8 handwritten-digits dataset a one-hidden-layer MLP reaches 96.2% test accuracy, identical to scikit-learn's.

## Complexity
Per epoch $O(n\sum_l n_{l-1}n_l)$; memory $O(\text{batch}\times\text{widest layer})$ activations + parameters.

## References
Rumelhart, Hinton & Williams (1986). Glorot & Bengio (2010). He et al. (2015). Srivastava et al. (2014) *Dropout.*
