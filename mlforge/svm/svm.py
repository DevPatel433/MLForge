"""Support Vector Machines: a primal linear SVM and a kernel SVM (SMO)."""
import numpy as np

from ..base import BaseEstimator, ClassifierMixin
from ..optimization import Adam, minimize_batches
from ..utils.pairwise import pairwise_distances


# ------------------------------------------------------------------- linear
def hinge_loss_grad(theta, X, t, C=1.0, loss="hinge", fit_intercept=True):
    """SVM primal objective in mean form and its (sub)gradient.

    L = mean(max(0, 1 - t f)^p) + ||w||^2 / (2 C n),   f = Xw + b,  t in {-1, +1}
    with p = 1 (hinge) or p = 2 (squared hinge).
    """
    n = X.shape[0]
    w, b = theta[:-1], theta[-1]
    margin = 1.0 - t * (X @ w + b)
    active = margin > 0
    if loss == "hinge":
        data_loss = np.mean(np.where(active, margin, 0.0))
        coef = np.where(active, -t, 0.0) / n
    elif loss == "squared_hinge":
        data_loss = np.mean(np.where(active, margin, 0.0) ** 2)
        coef = np.where(active, -2.0 * margin * t, 0.0) / n
    else:
        raise ValueError("loss must be 'hinge' or 'squared_hinge'")
    gw = X.T @ coef + w / (C * n)
    gb = coef.sum() if fit_intercept else 0.0
    return data_loss + w @ w / (2.0 * C * n), np.append(gw, gb)


class LinearSVC(BaseEstimator, ClassifierMixin):
    """Linear SVM trained by (sub)gradient descent on the primal objective.
    Multiclass problems use one-vs-rest.

    ``C`` is the usual inverse regularisation strength: larger C = harder margin.
    """

    def __init__(self, learning_rate=0.05, C=1.0, n_iters=1000, loss="hinge",
                 optimizer=None, batch_size=None, tol=0.0, fit_intercept=True,
                 random_state=None):
        self.learning_rate = learning_rate
        self.C = C
        self.n_iters = n_iters
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.random_state = random_state

    weights = property(lambda self: self.coef_.ravel() if self.coef_.shape[0] == 1 else self.coef_)
    bias = property(lambda self: float(self.intercept_[0]) if len(self.intercept_) == 1 else self.intercept_)
    cost_history = property(lambda self: self.cost_history_)

    def _fit_binary(self, X, t):
        n, d = X.shape
        opt = self.optimizer if self.optimizer is not None else Adam(self.learning_rate)
        theta, hist = minimize_batches(
            lambda th, idx: hinge_loss_grad(
                th, X if idx is None else X[idx], t if idx is None else t[idx],
                self.C, self.loss, self.fit_intercept),
            np.zeros(d + 1), opt, self.n_iters, n, self.batch_size, self.tol,
            self.random_state, learning_rate=self.learning_rate)
        return theta, hist

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        targets = [self.classes_[1]] if k == 2 else list(self.classes_)
        W, B, hists = [], [], []
        for c in targets:
            theta, h = self._fit_binary(X, np.where(y == c, 1.0, -1.0))
            W.append(theta[:-1])
            B.append(theta[-1])
            hists.append(h)
        self.coef_, self.intercept_ = np.array(W), np.array(B)
        self.cost_history_ = hists[0] if len(hists) == 1 else hists
        self.n_features_in_ = X.shape[1]
        return self

    def decision_function(self, X):
        s = X @ self.coef_.T + self.intercept_
        return s.ravel() if s.shape[1] == 1 else s

    def predict(self, X):
        s = self.decision_function(X)
        if s.ndim == 1:
            return self.classes_[(s > 0).astype(int)]
        return self.classes_[np.argmax(s, axis=1)]


# ------------------------------------------------------------------- kernel
def kernel_matrix(X, Y, kernel="rbf", gamma=1.0, degree=3, coef0=0.0):
    if callable(kernel):
        return kernel(X, Y)
    if kernel == "linear":
        return X @ Y.T
    if kernel == "poly":
        return (gamma * (X @ Y.T) + coef0) ** degree
    if kernel == "rbf":
        return np.exp(-gamma * pairwise_distances(X, Y, "sqeuclidean"))
    if kernel == "sigmoid":
        return np.tanh(gamma * (X @ Y.T) + coef0)
    raise ValueError(f"Unknown kernel {kernel!r}")


def smo(K, t, C, tol=1e-3, max_passes=10, max_iter=100000, rng=None):
    """Sequential Minimal Optimisation (Platt, 1998) for the SVM dual

        max  sum(a) - 1/2 sum_ij a_i a_j t_i t_j K_ij     0 <= a_i <= C,  sum a_i t_i = 0

    Uses the max-|E_i - E_j| second-choice heuristic with an error cache.
    Returns ``(alpha, b)``.
    """
    rng = rng or np.random.RandomState(0)
    n = len(t)
    alpha = np.zeros(n)
    b = 0.0
    f = np.zeros(n)                     # f_i = sum_j a_j t_j K_ij + b
    eps = 1e-12

    def take_step(i, j):
        nonlocal b
        if i == j:
            return False
        Ei, Ej = f[i] - t[i], f[j] - t[j]
        ai, aj = alpha[i], alpha[j]
        if t[i] != t[j]:
            L, H = max(0.0, aj - ai), min(C, C + aj - ai)
        else:
            L, H = max(0.0, ai + aj - C), min(C, ai + aj)
        if H - L < eps:
            return False
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        if eta <= eps:
            return False
        aj_new = np.clip(aj + t[j] * (Ei - Ej) / eta, L, H)
        if abs(aj_new - aj) < 1e-7 * (aj_new + aj + 1e-7):
            return False
        ai_new = ai + t[i] * t[j] * (aj - aj_new)
        b1 = b - Ei - t[i] * (ai_new - ai) * K[i, i] - t[j] * (aj_new - aj) * K[i, j]
        b2 = b - Ej - t[i] * (ai_new - ai) * K[i, j] - t[j] * (aj_new - aj) * K[j, j]
        b_new = b1 if 0 < ai_new < C else (b2 if 0 < aj_new < C else (b1 + b2) / 2)
        f[:] += (ai_new - ai) * t[i] * K[:, i] + (aj_new - aj) * t[j] * K[:, j] + (b_new - b)
        alpha[i], alpha[j], b = ai_new, aj_new, b_new
        return True

    passes, it = 0, 0
    while passes < max_passes and it < max_iter:
        changed = 0
        for i in range(n):
            it += 1
            Ei = f[i] - t[i]
            if (t[i] * Ei < -tol and alpha[i] < C) or (t[i] * Ei > tol and alpha[i] > 0):
                nb = np.where((alpha > 0) & (alpha < C))[0]
                tried = False
                if len(nb) > 1:
                    errs = np.abs(Ei - (f[nb] - t[nb]))
                    j = nb[int(np.argmax(errs))]
                    tried = take_step(i, j)
                if not tried:
                    for j in rng.permutation(n)[:min(n, 20)]:
                        if take_step(i, j):
                            tried = True
                            break
                changed += int(tried)
        passes = passes + 1 if changed == 0 else 0
    return alpha, b


class SVC(BaseEstimator, ClassifierMixin):
    """Kernel SVM classifier trained with SMO (one-vs-one for multiclass).

    kernel : 'linear', 'poly', 'rbf', 'sigmoid' or a callable ``k(X, Y)``.
    gamma  : 'scale' (``1 / (d * X.var())``), 'auto' (``1 / d``) or a float.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", degree=3, coef0=0.0,
                 tol=1e-3, max_passes=10, random_state=None):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_passes = max_passes
        self.random_state = random_state

    def _gamma(self, X):
        if self.gamma == "scale":
            v = X.var()
            return 1.0 / (X.shape[1] * v) if v > 0 else 1.0
        if self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def _K(self, A, B):
        return kernel_matrix(A, B, self.kernel, self.gamma_, self.degree, self.coef0)

    def _fit_pair(self, X, t, rng):
        K = self._K(X, X)
        alpha, b = smo(K, t, self.C, self.tol, self.max_passes, rng=rng)
        sv = alpha > 1e-8
        return {"sv": X[sv], "coef": (alpha * t)[sv], "b": b, "idx": np.where(sv)[0]}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.gamma_ = self._gamma(X)
        rng = np.random.RandomState(self.random_state)
        self._models, self._pairs = [], []
        k = len(self.classes_)
        for a in range(k):
            for c in range(a + 1, k):
                mask = (y == self.classes_[a]) | (y == self.classes_[c])
                t = np.where(y[mask] == self.classes_[c], 1.0, -1.0)
                m = self._fit_pair(X[mask], t, rng)
                m["idx"] = np.where(mask)[0][m["idx"]]
                self._models.append(m)
                self._pairs.append((a, c))
        sv_idx = np.unique(np.concatenate([m["idx"] for m in self._models]))
        self.support_ = sv_idx
        self.support_vectors_ = X[sv_idx]
        self.n_support_ = np.array([len(np.intersect1d(sv_idx, np.where(y == c)[0])) for c in self.classes_])
        if k == 2:
            self.dual_coef_ = self._models[0]["coef"][None, :]
            self.intercept_ = np.array([self._models[0]["b"]])
        self.n_features_in_ = X.shape[1]
        return self

    def _pair_scores(self, X):
        return np.column_stack([self._K(X, m["sv"]) @ m["coef"] + m["b"] for m in self._models])

    def decision_function(self, X):
        s = self._pair_scores(X)
        return s[:, 0] if len(self.classes_) == 2 else s

    def predict(self, X):
        s = self._pair_scores(X)
        k = len(self.classes_)
        if k == 2:
            return self.classes_[(s[:, 0] > 0).astype(int)]
        votes = np.zeros((len(X), k))
        conf = np.zeros((len(X), k))
        for col, (a, c) in enumerate(self._pairs):
            win_c = s[:, col] > 0
            votes[win_c, c] += 1
            votes[~win_c, a] += 1
            conf[:, c] += s[:, col]
            conf[:, a] -= s[:, col]
        return self.classes_[np.argmax(votes + 1e-3 * conf / (np.abs(conf).max() + 1e-12), axis=1)]
