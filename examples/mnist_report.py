"""Figures for the MNIST CNN, generated from the saved weights and the JSON log (no retraining).

    python examples/mnist_report.py          # -> docs/images/mnist_cnn_*.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mnist_cnn import build, evaluate  # noqa: E402

from mlforge import visualization as viz  # noqa: E402
from mlforge.datasets import load_mnist  # noqa: E402
from mlforge.metrics import classification_report  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "benchmarks", "results")
OUT = os.path.join(HERE, "..", "docs", "images")


def main(cache_dir=None):
    os.makedirs(OUT, exist_ok=True)
    log = json.load(open(os.path.join(RES, "mnist_cnn.json")))
    mlp_log = json.load(open(os.path.join(RES, "mnist_mlp.json")))
    _, _, Xte, yte = load_mnist(cache_dir=cache_dir)
    net = build("cnn", 0)
    net.set_flat_params(np.load(os.path.join(RES, "mnist_cnn_weights.npz"))["params"])
    acc, _ = evaluate(net, Xte, yte)
    assert abs(acc - log["final_test_accuracy"]) < 1e-12, "saved weights do not reproduce the logged accuracy"
    pred = np.concatenate([net.forward(Xte[s:s + 1000], training=False).argmax(1) for s in range(0, len(Xte), 1000)])

    # 1) training curves, CNN vs MLP
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, lg, c in (("CNN (5,258 params)", log, "#4C72B0"), ("MLP (109,386 params)", mlp_log, "#DD8452")):
        ep = [h["epoch"] for h in lg["history"]]
        ax[0].plot(ep, [h["test_loss"] for h in lg["history"]], "o-", color=c, label=name)
        ax[1].plot(ep, [h["test_acc"] for h in lg["history"]], "o-", color=c, label=name)
    ax[0].set(title="Test loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].set(title="Test accuracy (10,000 held-out images)", xlabel="epoch", ylabel="accuracy")
    ax[1].legend(frameon=False)
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle(f"MNIST, from scratch in NumPy - CNN {log['final_test_accuracy']:.2%} vs MLP {mlp_log['final_test_accuracy']:.2%}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mnist_cnn_training.png"), dpi=130)

    # 2) confusion matrix
    viz.plot_confusion_matrix(yte, pred, normalize="true", title=f"MNIST confusion matrix - CNN ({acc:.2%})",
                              save=os.path.join(OUT, "mnist_cnn_confusion.png"))

    # 3) learned first-layer filters + the most confident mistakes
    W = net.layers[0].W[:, 0]
    wrong = np.where(pred != yte)[0]
    conf = np.array([np.exp(net.forward(Xte[i:i + 1], training=False)).max() for i in wrong[:200]])
    show = wrong[:200][np.argsort(-conf)[:8]]
    fig, axes = plt.subplots(2, 8, figsize=(13, 3.6))
    for k in range(8):
        axes[0, k].imshow(W[k], cmap="RdBu_r", vmin=-np.abs(W).max(), vmax=np.abs(W).max())
        axes[0, k].set_title(f"filter {k}", fontsize=8)
        axes[1, k].imshow(Xte[show[k], 0], cmap="gray_r")
        axes[1, k].set_title(f"true {yte[show[k]]} / pred {pred[show[k]]}", fontsize=8)
    for a in axes.ravel():
        a.axis("off")
    fig.suptitle("Learned 3x3 first-layer filters (top) and mistakes the CNN was most sure about (bottom)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "mnist_cnn_filters_and_errors.png"), dpi=130)
    print(classification_report(yte, pred, digits=3))
    plt.close("all")


if __name__ == "__main__":
    main(os.environ.get("MNIST_CACHE"))
