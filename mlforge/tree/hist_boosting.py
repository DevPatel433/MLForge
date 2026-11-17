"""Histogram-based gradient boosting (the LightGBM / ``HistGradientBoosting`` family).

Two ideas make this faster than exact-split boosting, especially for many samples:

1. **Binning.** Each feature is discretised once into at most ``max_bins`` (<= 256) ordered bins,
   so a split search only looks at bin boundaries instead of every distinct value.
2. **Histograms.** For a tree node, the sums of gradients ``g``, hessians ``h`` and sample counts
   per (feature, bin) are accumulated with one ``np.bincount``. The best split of *every* feature
   then follows from cumulative sums, and a child's histogram is obtained by *subtracting* the
   sibling's from the parent's, so only the smaller child is ever scanned.

Trees are grown leaf-wise (best-first) up to ``max_leaf_nodes`` leaves. With second-order
(Newton) boosting a split has gain

    G_L^2/(H_L+l2) + G_R^2/(H_R+l2) - G^2/(H+l2)

and a leaf has value ``-G/(H+l2)`` (before shrinkage by ``learning_rate``).
"""
import heapq

import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin
from ..utils.math_utils import sigmoid, softmax

N_BINS = 256


class BinMapper:
    """Quantile / midpoint binning of each feature into at most ``max_bins`` ordered bins."""

    def __init__(self, max_bins=255):
        if not 2 <= max_bins <= N_BINS:
            raise ValueError("max_bins must be between 2 and 256 (bin ids are stored as uint8)")
        self.max_bins = max_bins

    def fit(self, X):
        self.thresholds_ = []
        for j in range(X.shape[1]):
            col = X[:, j]
            distinct = np.unique(col)
            if len(distinct) <= self.max_bins:
                th = (distinct[:-1] + distinct[1:]) / 2.0               # midpoints between neighbouring values
                th = np.where(th >= distinct[1:], distinct[:-1], th)    # 1-ulp neighbours: midpoint rounded up
            else:
                th = np.unique(np.percentile(col, np.linspace(0, 100, self.max_bins + 1)[1:-1]))
            self.thresholds_.append(th)
        self.n_bins_ = max(len(th) for th in self.thresholds_) + 1          # bins actually in use (<= 256)
        return self

    def transform(self, X):
        Xb = np.empty(X.shape, dtype=np.uint8)
        for j, th in enumerate(self.thresholds_):
            Xb[:, j] = np.searchsorted(th, X[:, j], side="left")        # value <= th[b]  <=>  bin <= b
        return Xb


def flat_bin_index(Xb, n_bins=N_BINS):
    """Row-wise ``feature * n_bins + bin`` ids so that one bincount covers all features."""
    return Xb.astype(np.intp) + (np.arange(Xb.shape[1]) * n_bins)[None, :]


def histograms(flat_rows, g_rows, h_rows, n_features, n_bins=N_BINS):
    """Sums of g, h and counts for every (feature, bin) of the given rows."""
    flat = flat_rows.ravel()
    size = n_features * n_bins
    hg = np.bincount(flat, weights=np.repeat(g_rows, n_features), minlength=size).reshape(n_features, n_bins)
    hh = np.bincount(flat, weights=np.repeat(h_rows, n_features), minlength=size).reshape(n_features, n_bins)
    hc = np.bincount(flat, minlength=size).reshape(n_features, n_bins).astype(float)
    return hg, hh, hc


def best_split_from_histograms(hg, hh, hc, G, H, C, l2, min_samples_leaf):
    """Best ``(feature, bin, gain)`` over all features, from per-bin sums (see module docstring)."""
    cg, ch, cc = (np.cumsum(a, axis=1)[:, :-1] for a in (hg, hh, hc))
    rg, rh, rc = G - cg, H - ch, C - cc
    valid = (cc >= min_samples_leaf) & (rc >= min_samples_leaf)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = cg ** 2 / (ch + l2 + 1e-12) + rg ** 2 / (rh + l2 + 1e-12) - G ** 2 / (H + l2 + 1e-12)
    gain = np.where(valid, gain, -np.inf)
    flat = int(np.argmax(gain))
    f, b = divmod(flat, gain.shape[1])
    return f, b, float(gain[f, b])


class HistTree:
    """One regression tree grown leaf-wise on binned data from gradients and hessians."""

    def __init__(self, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2=0.0, min_gain=1e-9, n_bins=N_BINS):
        self.max_leaf_nodes, self.max_depth = max_leaf_nodes, max_depth
        self.min_samples_leaf, self.l2, self.min_gain, self.n_bins = min_samples_leaf, l2, min_gain, n_bins

    def _leaf_value(self, G, H):
        return -G / (H + self.l2 + 1e-12)

    def fit(self, Xb, g, h, flat=None):
        d = Xb.shape[1]
        flat = flat_bin_index(Xb, self.n_bins) if flat is None else flat
        self.feature, self.bin, self.left, self.right, self.value = [], [], [], [], []
        self.gain_per_feature = np.zeros(d)

        def new_node(G, H):
            self.feature.append(-1)
            self.bin.append(0)
            self.left.append(-1)
            self.right.append(-1)
            self.value.append(self._leaf_value(G, H))
            return len(self.value) - 1

        def candidate(node, rows, hist, depth):
            G, H, C = g[rows].sum(), h[rows].sum(), float(len(rows))
            if (self.max_depth is not None and depth >= self.max_depth) or C < 2 * self.min_samples_leaf:
                return None
            f, b, gain = best_split_from_histograms(*hist, G, H, C, self.l2, self.min_samples_leaf)
            return (-gain, node, rows, hist, depth, f, b) if gain > self.min_gain else None

        rows = np.arange(len(g))
        root = new_node(g.sum(), h.sum())
        heap, counter = [], 0
        c = candidate(root, rows, histograms(flat, g, h, d, self.n_bins), 0)
        if c:
            heapq.heappush(heap, (c[0], counter, c))
        n_leaves = 1
        while heap and n_leaves < self.max_leaf_nodes:
            _, _, (neg_gain, node, rows, hist, depth, f, b) = heapq.heappop(heap)
            go_left = Xb[rows, f] <= b
            lr, rr = rows[go_left], rows[~go_left]
            small, large = (lr, rr) if len(lr) <= len(rr) else (rr, lr)
            h_small = histograms(flat[small], g[small], h[small], d, self.n_bins)
            h_large = tuple(p - s for p, s in zip(hist, h_small))          # histogram subtraction trick
            hist_l, hist_r = (h_small, h_large) if small is lr else (h_large, h_small)
            left = new_node(g[lr].sum(), h[lr].sum())
            right = new_node(g[rr].sum(), h[rr].sum())
            self.feature[node], self.bin[node], self.left[node], self.right[node] = f, b, left, right
            self.gain_per_feature[f] += -neg_gain
            n_leaves += 1
            for child, crow, chist in ((left, lr, hist_l), (right, rr, hist_r)):
                c = candidate(child, crow, chist, depth + 1)
                if c:
                    counter += 1
                    heapq.heappush(heap, (c[0], counter, c))
        self.feature, self.bin = np.array(self.feature), np.array(self.bin)
        self.left, self.right, self.value = np.array(self.left), np.array(self.right), np.array(self.value)
        return self

    @property
    def n_leaves(self):
        return int((self.left == -1).sum())

    def predict(self, Xb):
        node = np.zeros(len(Xb), dtype=np.intp)
        active = self.left[node] != -1
        while active.any():
            idx = np.where(active)[0]
            cur = node[idx]
            go_left = Xb[idx, self.feature[cur]] <= self.bin[cur]
            node[idx] = np.where(go_left, self.left[cur], self.right[cur])
            active = self.left[node] != -1
        return self.value[node]


class _BaseHistGB(BaseEstimator):
    def __init__(self, max_iter=100, learning_rate=0.1, max_leaf_nodes=31, max_depth=None,
                 min_samples_leaf=20, l2_regularization=0.0, max_bins=255):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.max_bins = max_bins

    def _new_tree(self):
        return HistTree(self.max_leaf_nodes, self.max_depth, self.min_samples_leaf, self.l2_regularization,
                        n_bins=self.bin_mapper_.n_bins_)

    def _prepare(self, X):
        self.bin_mapper_ = BinMapper(self.max_bins).fit(X)
        self.n_features_in_ = X.shape[1]
        Xb = self.bin_mapper_.transform(X)
        return Xb, flat_bin_index(Xb, self.bin_mapper_.n_bins_)

    def _finish(self):
        imp = np.sum([t.gain_per_feature for trees in self.estimators_ for t in trees], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp

    def _raw(self, X):
        Xb = self.bin_mapper_.transform(X)
        F = np.tile(self.init_, (len(X), 1))
        for trees in self.estimators_:
            for k, t in enumerate(trees):
                F[:, k] += self.learning_rate * t.predict(Xb)
        return F


class HistGradientBoostingRegressor(_BaseHistGB, RegressorMixin):
    """Histogram-based gradient boosting with squared loss (``g = F - y``, ``h = 1``)."""

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        Xb, flat = self._prepare(X)
        self.init_ = np.array([y.mean()])
        F = np.full(len(y), self.init_[0])
        ones = np.ones(len(y))
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.max_iter):
            tree = self._new_tree().fit(Xb, F - y, ones, flat)
            F += self.learning_rate * tree.predict(Xb)
            self.estimators_.append([tree])
            self.train_loss_.append(float(np.mean((y - F) ** 2)))
        self._finish()
        return self

    def predict(self, X):
        return self._raw(X)[:, 0]

    def staged_predict(self, X):
        Xb = self.bin_mapper_.transform(X)
        F = np.full(len(X), self.init_[0])
        for trees in self.estimators_:
            F = F + self.learning_rate * trees[0].predict(Xb)
            yield F.copy()


class HistGradientBoostingClassifier(_BaseHistGB, ClassifierMixin):
    """Histogram-based boosting with the (binary or multinomial) log-loss.

    Binary: one tree per iteration with ``g = p - y``, ``h = p(1-p)``.
    Multiclass: one tree per class per iteration on the softmax gradients.
    """

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        Xb, flat = self._prepare(X)
        n = len(y_idx)
        Y = np.eye(k)[y_idx]
        if k == 2:
            p0 = np.clip(Y[:, 1].mean(), 1e-6, 1 - 1e-6)
            self.init_ = np.array([np.log(p0 / (1 - p0))])
            F = np.full((n, 1), self.init_[0])
        else:
            self.init_ = np.log(np.clip(Y.mean(0), 1e-6, None))
            F = np.tile(self.init_, (n, 1))
        self.estimators_, self.train_loss_ = [], []
        for _ in range(self.max_iter):
            P = sigmoid(F) if k == 2 else softmax(F, axis=1)
            trees = []
            for c in range(F.shape[1]):
                pc = P[:, c]
                target = Y[:, 1] if k == 2 else Y[:, c]
                tree = self._new_tree().fit(Xb, pc - target, np.maximum(pc * (1 - pc), 1e-12), flat)
                F[:, c] += self.learning_rate * tree.predict(Xb)
                trees.append(tree)
            self.estimators_.append(trees)
            P = sigmoid(F) if k == 2 else softmax(F, axis=1)
            if k == 2:
                self.train_loss_.append(float(-np.mean(Y[:, 1] * np.log(P[:, 0] + 1e-12) + (1 - Y[:, 1]) * np.log(1 - P[:, 0] + 1e-12))))
            else:
                self.train_loss_.append(float(-np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))))
        self._finish()
        return self

    def decision_function(self, X):
        F = self._raw(X)
        return F[:, 0] if len(self.classes_) == 2 else F

    def predict_proba(self, X):
        F = self._raw(X)
        if len(self.classes_) == 2:
            p = sigmoid(F[:, 0])
            return np.column_stack([1 - p, p])
        return softmax(F, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
