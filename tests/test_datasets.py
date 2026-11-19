import numpy as np
import pytest

from mlforge import datasets as D


@pytest.mark.parametrize("fn,kw", [
    (D.make_blobs, dict(n_samples=50, centers=4)), (D.make_moons, dict(n_samples=51)),
    (D.make_circles, dict(n_samples=51)), (D.make_spirals, dict(n_samples=60, n_classes=3)),
    (D.make_classification, dict(n_samples=50, n_classes=3))])
def test_generators_shapes_and_determinism(fn, kw):
    X, y = fn(random_state=3, **kw)
    X2, y2 = fn(random_state=3, **kw)
    assert len(X) == len(y) and np.array_equal(X, X2) and np.array_equal(y, y2)
    assert not np.array_equal(X, fn(random_state=4, **kw)[0])


def test_blobs_with_given_centers_and_regression():
    X, y = D.make_blobs(200, [[0, 0], [10, 10]], cluster_std=0.1, random_state=0)
    assert np.allclose(X[y == 1].mean(0), [10, 10], atol=0.1)
    X, y, w = D.make_regression(100, 3, noise=0.0, coef=True, bias=2.0, random_state=0)
    assert np.allclose(X @ w + 2.0, y)
    assert D.make_regression(10, 1, random_state=0)[0].shape == (10, 1)


# ------------------------------------------------------------------------------------- MNIST loader (offline)
def _idx(arr):
    arr = np.asarray(arr, dtype=np.uint8)
    header = (0x0800 + arr.ndim).to_bytes(4, "big") + b"".join(int(d).to_bytes(4, "big") for d in arr.shape)
    return header + arr.tobytes()


def test_parse_idx_round_trip_and_errors():
    from mlforge.datasets.mnist import parse_idx
    imgs = np.random.RandomState(0).randint(0, 256, (5, 4, 3))
    assert np.array_equal(parse_idx(_idx(imgs)), imgs)
    assert np.array_equal(parse_idx(_idx(np.arange(7))), np.arange(7))
    with pytest.raises(ValueError, match="truncated"):
        parse_idx(_idx(imgs)[:-3])
    with pytest.raises(ValueError, match="unsupported"):
        parse_idx((0x0D01).to_bytes(4, "big") + (1).to_bytes(4, "big") + b"\0")


def test_mnist_loader_verifies_checksums_and_prefers_the_cache(tmp_path, monkeypatch):
    import gzip
    import hashlib

    import mlforge.datasets.mnist as mn
    files = {"train-images-idx3-ubyte.gz": _idx(np.random.RandomState(1).randint(0, 256, (6, 28, 28))),
             "train-labels-idx1-ubyte.gz": _idx(np.arange(6)),
             "t10k-images-idx3-ubyte.gz": _idx(np.random.RandomState(2).randint(0, 256, (3, 28, 28))),
             "t10k-labels-idx1-ubyte.gz": _idx(np.arange(3))}
    blobs = {k: gzip.compress(v) for k, v in files.items()}
    monkeypatch.setattr(mn, "SHA256", {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()})
    served = []

    class FakeResponse:
        def __init__(self, data): self.data = data
        def read(self): return self.data
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(url, timeout=0):
        served.append(url)
        return FakeResponse(blobs[url.rsplit("/", 1)[1]])
    monkeypatch.setattr(mn.urllib.request, "urlopen", fake_urlopen)
    Xtr, ytr, Xte, yte = D.load_mnist(cache_dir=str(tmp_path))
    assert Xtr.shape == (6, 1, 28, 28) and Xtr.dtype == np.float32 and 0 <= Xtr.min() and Xtr.max() <= 1
    assert ytr.tolist() == list(range(6)) and Xte.shape == (3, 1, 28, 28) and len(served) == 4
    assert D.load_mnist(cache_dir=str(tmp_path), flatten=True, normalize=False)[0].shape == (6, 784)
    assert len(served) == 4                                             # second call is served from the cache
    # a corrupted cached file is detected and re-downloaded
    (tmp_path / "train-labels-idx1-ubyte.gz").write_bytes(b"garbage")
    D.load_mnist(cache_dir=str(tmp_path))
    assert len(served) == 5


def test_mnist_loader_rejects_a_tampered_download_and_reports_all_mirrors(tmp_path, monkeypatch):
    import mlforge.datasets.mnist as mn

    class Evil:
        def read(self): return b"not mnist"
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(mn.urllib.request, "urlopen", lambda url, timeout=0: Evil())
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        D.load_mnist(cache_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.gz"))                              # nothing untrusted is ever written to the cache

    def offline(url, timeout=0):
        raise OSError("no network")
    monkeypatch.setattr(mn.urllib.request, "urlopen", offline)
    with pytest.raises(RuntimeError, match="OSError"):
        D.load_mnist(cache_dir=str(tmp_path))
