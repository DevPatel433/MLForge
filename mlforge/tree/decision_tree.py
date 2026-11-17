"""CART decision trees (classification and regression).

Split search is vectorised: for each candidate feature the samples are sorted
once and impurities of *all* thresholds are computed from cumulative class
counts (or cumulative sums for regression), giving ``O(d n log n)`` per node
instead of the naive ``O(d n^2)``.
"""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin, RegressorMixin


class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "impurity", "n_samples", "gain")

    def __init__(self, value, impurity, n_samples):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value
        self.impurity = impurity
        self.n_samples = n_samples
        self.gain = 0.0

    @property
    def is_leaf(self):
        return self.left is None


def _gini(counts):
    tot = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(tot, 1)
    return 1.0 - np.sum(p ** 2, axis=-1)


def _entropy(counts):
    tot = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(tot, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
    return -np.sum(p * logp, axis=-1)


class _BaseTree(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_impurity_decrease=0.0, max_features=None, ccp_alpha=0.0,
                 random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.ccp_alpha = ccp_alpha
        self.random_state = random_state

    # ---------------------------------------------------------------- building
    def _n_features_split(self, d):
        mf = self.max_features
        if mf is None:
            return d
        if mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        if mf == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(mf, float):
            return max(1, int(mf * d))
        return min(int(mf), d)

    def _grow(self, X, y, sample_weight=None):
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.RandomState(self.random_state)
        self._k_features = self._n_features_split(X.shape[1])
        self._n_total = len(y)
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root_ = self._build(X, y, 0)
        if self.ccp_alpha > 0:
            self._prune(self.ccp_alpha)
        self._recompute_importances()
        del self._rng

    def _build(self, X, y, depth):
        node = Node(self._leaf_value(y), self._node_impurity(y), len(y))
        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or len(y) < self.min_samples_split
            or len(y) < 2 * self.min_samples_leaf
            or node.impurity <= 1e-15
        )
        if stop:
            return node
        feat, thr, gain = self._best_split(X, y, node.impurity)
        if feat is None or gain * len(y) / self._n_total + 1e-12 < self.min_impurity_decrease:
            return node
        mask = X[:, feat] <= thr
        if mask.all() or not mask.any():        # defensive: a split must always partition the node
            return node
        node.feature, node.threshold, node.gain = feat, thr, gain
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def _best_split(self, X, y, parent_impurity):
        """Best (feature, threshold, gain), evaluating *all* thresholds of *all*
        candidate features with array operations (no Python loop over features)."""
        n, d = X.shape
        feats = np.arange(d)
        if self._k_features < d:
            feats = self._rng.choice(d, self._k_features, replace=False)
        msl = self.min_samples_leaf
        best = (None, None, -np.inf)
        for chunk in self._feature_chunks(feats, n):
            Xf = X[:, chunk]
            order = np.argsort(Xf, axis=0, kind="mergesort")
            xs = np.take_along_axis(Xf, order, 0)
            valid = xs[1:] != xs[:-1]                                   # (n-1, f)
            if not valid.any():
                continue
            left_imp, right_imp, n_left = self._split_impurities(y[order])   # (n-1, f), (n-1, f), (n-1,)
            n_left = n_left[:, None]
            n_right = n - n_left
            ok = valid & (n_left >= msl) & (n_right >= msl)
            gains = np.where(ok, parent_impurity - (n_left * left_imp + n_right * right_imp) / n, -np.inf)
            rows = np.argmax(gains, axis=0)                              # best threshold per feature
            per_feat = gains[rows, np.arange(len(chunk))]
            k = int(np.argmax(per_feat))                                 # first feature wins ties
            g = per_feat[k]
            if np.isfinite(g) and g > best[2] + 1e-15:
                i = rows[k]
                lo, hi = xs[i, k], xs[i + 1, k]
                thr = (lo + hi) / 2.0
                if not lo <= thr < hi:      # neighbours one ulp apart: the midpoint rounds up to `hi` (or overflows)
                    thr = lo                # `x <= lo` still separates them (same fallback as scikit-learn)
                best = (int(chunk[k]), float(thr), float(g))
        return best

    def _feature_chunks(self, feats, n):
        """Split the candidate features so the (n x f x classes) work array stays small."""
        width = max(1, int(4e6 // max(1, n * getattr(self, "_n_classes", 1))))
        for s in range(0, len(feats), width):
            yield feats[s:s + width]

    # ------------------------------------------------------------------ pruning
    def _subtree_stats(self, node):
        if node.is_leaf:
            return 1, node.impurity * node.n_samples / self._n_total
        nl, rl = self._subtree_stats(node.left)
        nr, rr = self._subtree_stats(node.right)
        return nl + nr, rl + rr

    def _prune(self, alpha):
        """Minimal cost-complexity pruning (Breiman et al., 1984): repeatedly
        collapse the weakest link while its effective alpha <= ``alpha``."""
        while True:
            weakest, best_g = None, np.inf
            stack = [self.root_]
            while stack:
                nd = stack.pop()
                if nd.is_leaf:
                    continue
                n_leaves, r_sub = self._subtree_stats(nd)
                r_node = nd.impurity * nd.n_samples / self._n_total
                g = (r_node - r_sub) / (n_leaves - 1)
                if g < best_g:
                    best_g, weakest = g, nd
                stack += [nd.left, nd.right]
            if weakest is None or best_g > alpha:
                return
            weakest.left = weakest.right = None
            weakest.feature = weakest.threshold = None

    def cost_complexity_pruning_path(self, X, y):
        """Effective alphas at which subtrees would be pruned."""
        import copy
        tree = copy.deepcopy(self).fit(X, y)
        alphas = [0.0]
        while True:
            weakest, best_g = None, np.inf
            stack = [tree.root_]
            while stack:
                nd = stack.pop()
                if nd.is_leaf:
                    continue
                n_leaves, r_sub = tree._subtree_stats(nd)
                g = (nd.impurity * nd.n_samples / tree._n_total - r_sub) / (n_leaves - 1)
                if g < best_g:
                    best_g, weakest = g, nd
                stack += [nd.left, nd.right]
            if weakest is None:
                return np.array(alphas)
            alphas.append(max(best_g, alphas[-1]))
            weakest.left = weakest.right = None

    def _recompute_importances(self):
        imp = np.zeros(self.n_features_in_)
        stack = [self.root_]
        while stack:
            nd = stack.pop()
            if nd.is_leaf:
                continue
            imp[nd.feature] += nd.gain * nd.n_samples
            stack += [nd.left, nd.right]
        total = imp.sum()
        self.feature_importances_ = imp / total if total > 0 else imp

    # -------------------------------------------------------------- inspection
    def get_depth(self):
        def d(nd):
            return 0 if nd.is_leaf else 1 + max(d(nd.left), d(nd.right))
        return d(self.root_)

    def get_n_leaves(self):
        return self._subtree_stats(self.root_)[0]

    def node_count(self):
        def c(nd):
            return 1 if nd.is_leaf else 1 + c(nd.left) + c(nd.right)
        return c(self.root_)

    def export_text(self, feature_names=None, decimals=3):
        """Human readable rules of the learned tree."""
        names = feature_names or [f"x[{i}]" for i in range(self.n_features_in_)]
        lines = []

        def rec(nd, depth):
            pad = "|   " * depth
            if nd.is_leaf:
                lines.append(f"{pad}|--- {self._describe_leaf(nd, decimals)}")
                return
            lines.append(f"{pad}|--- {names[nd.feature]} <= {nd.threshold:.{decimals}f}")
            rec(nd.left, depth + 1)
            lines.append(f"{pad}|--- {names[nd.feature]} >  {nd.threshold:.{decimals}f}")
            rec(nd.right, depth + 1)
        rec(self.root_, 0)
        return "\n".join(lines)

    def _leaf_for(self, X):
        """Return the leaf Node reached by each row of X."""
        leaves = np.empty(len(X), dtype=object)

        def rec(nd, idx):
            if nd.is_leaf or len(idx) == 0:
                leaves[idx] = nd
                return
            mask = X[idx, nd.feature] <= nd.threshold
            rec(nd.left, idx[mask])
            rec(nd.right, idx[~mask])
        rec(self.root_, np.arange(len(X)))
        return leaves


class DecisionTreeClassifier(_BaseTree, ClassifierMixin):
    """Classification tree using ``criterion='gini'`` or ``'entropy'``
    (information gain). Pass ``ccp_alpha > 0`` for cost-complexity pruning."""

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini",
                 min_samples_leaf=1, min_impurity_decrease=0.0, max_features=None,
                 ccp_alpha=0.0, random_state=None):
        super().__init__(max_depth, min_samples_split, min_samples_leaf,
                         min_impurity_decrease, max_features, ccp_alpha, random_state)
        self.criterion = criterion

    @property
    def root(self):
        return self.root_

    def _impurity_fn(self):
        if self.criterion == "gini":
            return _gini
        if self.criterion == "entropy":
            return _entropy
        raise ValueError("criterion must be 'gini' or 'entropy'")

    def _leaf_value(self, y):
        return np.bincount(y, minlength=self._n_classes).astype(float)

    def _node_impurity(self, y):
        return float(self._impurity_fn()(np.bincount(y, minlength=self._n_classes).astype(float)))

    def _split_impurities(self, ys):
        """``ys`` is (n, f): labels sorted along each feature. Returns the left and
        right impurities for every split position and the left sample counts."""
        n, f = ys.shape
        onehot = (ys[:, :, None] == np.arange(self._n_classes)).astype(float)       # (n, f, k)
        left = np.cumsum(onehot, axis=0)[:-1]
        right = left[-1:] * 0 + onehot.sum(0, keepdims=True) - left
        fn = self._impurity_fn()
        return fn(left), fn(right), left[:, 0].sum(-1)

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self._n_classes = len(self.classes_)
        self._grow(X, y_idx)
        return self

    def predict_proba(self, X):
        leaves = self._leaf_for(X)
        counts = np.array([nd.value for nd in leaves])
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def _describe_leaf(self, nd, decimals):
        return f"class: {self.classes_[int(np.argmax(nd.value))]}"


class DecisionTreeRegressor(_BaseTree, RegressorMixin):
    """Regression tree minimising the within-node variance (MSE)."""

    def _leaf_value(self, y):
        return float(np.mean(y))

    def _node_impurity(self, y):
        return float(np.var(y))

    def _split_impurities(self, ys):
        n = ys.shape[0]
        cs = np.cumsum(ys, axis=0)[:-1]
        cs2 = np.cumsum(ys ** 2, axis=0)[:-1]
        nl = np.arange(1, n, dtype=float)
        nr = (n - nl)[:, None]
        tot, tot2 = ys.sum(0), (ys ** 2).sum(0)
        nl_col = nl[:, None]
        left = cs2 / nl_col - (cs / nl_col) ** 2
        right = (tot2 - cs2) / nr - ((tot - cs) / nr) ** 2
        return np.maximum(left, 0), np.maximum(right, 0), nl

    def fit(self, X, y):
        self._grow(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return np.array([nd.value for nd in self._leaf_for(X)], dtype=float)

    def _describe_leaf(self, nd, decimals):
        return f"value: {nd.value:.{decimals}f}"
