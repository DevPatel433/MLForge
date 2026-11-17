"""Train neural networks on the REAL 60,000-image MNIST dataset, from scratch in NumPy.

    python examples/mnist_cnn.py --model cnn --epochs 5        # ~35 s / epoch on one CPU core
    python examples/mnist_cnn.py --model mlp --epochs 10
    python examples/mnist_cnn.py --model cnn --eval-only       # evaluate the saved weights

Every epoch's loss, accuracy and wall-clock time are written to ``benchmarks/results/mnist_<model>.json``
together with the exact configuration, so the numbers in the README can be reproduced and audited.
MNIST is downloaded once, checksum-verified and cached (see ``mlforge.datasets.load_mnist``).
"""
import argparse
import json
import os
import platform
import time

import numpy as np

import mlforge
from mlforge.datasets import load_mnist
from mlforge.neural_network import Conv2D, CrossEntropyLoss, Dense, Dropout, Flatten, MaxPool2D, ReLU, Sequential
from mlforge.optimization import get_optimizer

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks", "results")


def build(model, seed):
    if model == "cnn":        # 28x28 -> conv3x3(8) -> pool -> conv3x3(16) -> pool -> dense(10)   (5,258 parameters)
        return Sequential([Conv2D(1, 8, 3, random_state=seed), ReLU(), MaxPool2D(2),
                           Conv2D(8, 16, 3, random_state=seed + 1), ReLU(), MaxPool2D(2),
                           Flatten(), Dense(400, 10, "he", seed + 2)])
    return Sequential([Flatten(), Dense(784, 128, "he", seed), ReLU(), Dropout(0.1, seed), Dense(128, 64, "he", seed + 1), ReLU(),
                       Dense(64, 10, "he", seed + 2)])


def evaluate(net, X, y, chunk=1000):
    correct, loss, ce = 0, 0.0, CrossEntropyLoss()
    for s in range(0, len(X), chunk):
        out = net.forward(X[s:s + chunk], training=False)
        correct += int((out.argmax(1) == y[s:s + chunk]).sum())
        loss += ce.forward(out, y[s:s + chunk]) * len(out)
    return correct / len(X), loss / len(X)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=None, help="use only the first N training images (quick tests)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)

    Xtr, ytr, Xte, yte = load_mnist(cache_dir=args.cache_dir)
    if args.subset:
        Xtr, ytr = Xtr[:args.subset], ytr[:args.subset]
    net = build(args.model, args.seed)
    os.makedirs(args.out, exist_ok=True)
    weights_path = os.path.join(args.out, f"mnist_{args.model}_weights.npz")
    print(net.summary())

    if args.eval_only:
        data = np.load(weights_path)
        net.set_flat_params(data["params"])
        acc, loss = evaluate(net, Xte, yte)
        print(f"test accuracy {acc:.4f}  loss {loss:.4f}  ({len(Xte)} images)")
        return acc

    rng = np.random.RandomState(args.seed)
    loss_fn = CrossEntropyLoss()
    opts = [get_optimizer("adam", args.lr) for _ in net.parameters()]
    history, t_train = [], 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        order = rng.permutation(len(Xtr))
        total, correct = 0.0, 0
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            out = net.forward(Xtr[idx], training=True)
            total += loss_fn.forward(out, ytr[idx]) * len(idx)
            correct += int((out.argmax(1) == ytr[idx]).sum())
            net.backward(loss_fn.backward())
            for p, g, o in zip(net.parameters(), net.gradients(), opts):
                p -= o.step(g)
        dt = time.perf_counter() - t0
        t_train += dt
        te_acc, te_loss = evaluate(net, Xte, yte)
        history.append({"epoch": epoch, "train_loss": total / len(Xtr), "train_acc_running": correct / len(Xtr),
                        "test_loss": te_loss, "test_acc": te_acc, "epoch_seconds": round(dt, 1)})
        print(f"epoch {epoch}/{args.epochs}  train loss {total / len(Xtr):.4f}  test acc {te_acc:.4f}  test loss {te_loss:.4f}  ({dt:.0f}s)", flush=True)

    np.savez_compressed(weights_path, params=net.get_flat_params())
    result = {"model": args.model, "dataset": "MNIST (60,000 train / 10,000 test)", "n_train": len(Xtr), "n_test": len(Xte),
              "parameters": net.n_parameters(), "optimizer": "adam", "learning_rate": args.lr, "batch_size": args.batch_size,
              "epochs": args.epochs, "seed": args.seed, "final_test_accuracy": history[-1]["test_acc"],
              "total_training_seconds": round(t_train, 1), "history": history,
              "environment": {"mlforge": mlforge.__version__, "numpy": np.__version__, "python": platform.python_version(),
                              "machine": platform.machine(), "cpu_threads_available": os.cpu_count()}}
    with open(os.path.join(args.out, f"mnist_{args.model}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"final test accuracy {result['final_test_accuracy']:.4f} after {t_train:.0f}s of training  ->  {args.out}")
    return result


if __name__ == "__main__":
    main()
