"""MNIST loader: download once, verify SHA-256, cache, parse the IDX format with NumPy.

Uses only the standard library (``urllib``, ``gzip``, ``hashlib``) and NumPy. Files are verified
against hard-coded SHA-256 digests, so a corrupted or tampered mirror is rejected.

    from mlforge.datasets import load_mnist
    Xtr, ytr, Xte, yte = load_mnist()            # float32 in [0, 1], shape (N, 1, 28, 28)
"""
import gzip
import hashlib
import os
import urllib.request

import numpy as np

MIRRORS = (
    "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def default_cache_dir():
    return os.environ.get("MLFORGE_DATA", os.path.join(os.path.expanduser("~"), ".cache", "mlforge"))


def parse_idx(raw):
    """Parse the IDX byte format (big-endian header) into a uint8 array."""
    magic = int.from_bytes(raw[:4], "big")
    dtype_code, ndim = (magic >> 8) & 0xFF, magic & 0xFF
    if dtype_code != 0x08:
        raise ValueError(f"unsupported IDX data type 0x{dtype_code:02x} (expected unsigned byte)")
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError("IDX file is truncated or corrupted")
    return data.reshape(shape)


def _fetch(name, cache_dir, mirrors=MIRRORS, timeout=60):
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        if hashlib.sha256(open(path, "rb").read()).hexdigest() == SHA256[name]:
            return path
        os.remove(path)                                            # cached file is corrupt: re-download
    os.makedirs(cache_dir, exist_ok=True)
    errors = []
    for base in mirrors:
        try:
            with urllib.request.urlopen(base + name, timeout=timeout) as resp:
                data = resp.read()
        except Exception as exc:                                   # network errors differ widely: try the next mirror
            errors.append(f"{base}: {type(exc).__name__}")
            continue
        if hashlib.sha256(data).hexdigest() != SHA256[name]:
            errors.append(f"{base}: checksum mismatch")
            continue
        with open(path, "wb") as fh:
            fh.write(data)
        return path
    raise RuntimeError(f"could not obtain {name} with a valid checksum ({'; '.join(errors)}). "
                       f"Download it manually into {cache_dir}.")


def load_mnist(cache_dir=None, flatten=False, normalize=True, dtype=np.float32):
    """Return ``(X_train, y_train, X_test, y_test)`` - 60,000 / 10,000 handwritten digits.

    ``flatten=True`` gives (N, 784) instead of (N, 1, 28, 28); ``normalize`` scales to [0, 1].
    """
    cache_dir = cache_dir or default_cache_dir()
    out = []
    for stem in ("train", "t10k"):
        images = parse_idx(gzip.decompress(open(_fetch(f"{stem}-images-idx3-ubyte.gz", cache_dir), "rb").read()))
        labels = parse_idx(gzip.decompress(open(_fetch(f"{stem}-labels-idx1-ubyte.gz", cache_dir), "rb").read()))
        X = images.astype(dtype)
        if normalize:
            X /= 255.0
        X = X.reshape(len(X), -1) if flatten else X[:, None, :, :]
        out += [X, labels.astype(np.int64)]
    return tuple(out)
