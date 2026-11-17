"""Data splitting: hold-out split and cross-validation iterators."""
import numpy as np


def train_test_split(X, y=None, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    """Split arrays into train and test subsets (optionally stratified)."""
    X = np.asarray(X)
    n = X.shape[0]
    if y is not None and len(y) != n:
        raise ValueError(f"X and y have inconsistent lengths: {n} vs {len(y)}")
    if isinstance(test_size, float):
        if not 0 < test_size < 1:
            raise ValueError("test_size as a float must lie strictly between 0 and 1.")
        n_test = int(np.ceil(n * test_size))
    else:
        n_test = int(test_size)
    if not 0 < n_test < n:
        raise ValueError(f"test_size={test_size} leaves an empty train or test set (n={n}).")

    rng = np.random.RandomState(random_state)
    if stratify is not None:
        stratify = np.asarray(stratify)
        test_idx = []
        for c in np.unique(stratify):
            idx = np.where(stratify == c)[0]
            if shuffle:
                rng.shuffle(idx)
            test_idx.extend(idx[: max(1, int(round(len(idx) * n_test / n)))])
        test_idx = np.array(sorted(test_idx))
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        if shuffle:
            rng.shuffle(train_idx)
            rng.shuffle(test_idx)
    else:
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)
        test_idx, train_idx = idx[:n_test], idx[n_test:]

    if y is None:
        return X[train_idx], X[test_idx]
    y = np.asarray(y)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, X, y=None):
        n = len(X)
        if self.n_splits > n:
            raise ValueError(f"Cannot have n_splits={self.n_splits} > n_samples={n}.")
        idx = np.arange(n)
        if self.shuffle:
            np.random.RandomState(self.random_state).shuffle(idx)
        sizes = np.full(self.n_splits, n // self.n_splits)
        sizes[: n % self.n_splits] += 1
        start = 0
        for s in sizes:
            test = idx[start:start + s]
            yield np.concatenate([idx[:start], idx[start + s:]]), test
            start += s

    def get_n_splits(self, *args):
        return self.n_splits


class StratifiedKFold(KFold):
    """K-fold that preserves the class proportions in every fold."""

    def split(self, X, y):
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if self.shuffle:
                rng.shuffle(idx)
            for i, j in enumerate(idx):
                folds[i % self.n_splits].append(j)
        all_idx = np.arange(len(y))
        for f in folds:
            test = np.array(sorted(f))
            yield np.setdiff1d(all_idx, test), test


class ShuffleSplit:
    def __init__(self, n_splits=10, test_size=0.2, random_state=None):
        self.n_splits, self.test_size, self.random_state = n_splits, test_size, random_state

    def split(self, X, y=None):
        n = len(X)
        rng = np.random.RandomState(self.random_state)
        n_test = int(np.ceil(n * self.test_size))
        for _ in range(self.n_splits):
            p = rng.permutation(n)
            yield p[n_test:], p[:n_test]

    def get_n_splits(self, *args):
        return self.n_splits
