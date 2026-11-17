"""Classification metrics (binary and multiclass)."""
import numpy as np


def _labels(y_true, y_pred, labels=None):
    if labels is not None:
        return np.asarray(labels)
    return np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))


def accuracy_score(y_true, y_pred, normalize=True):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    hits = np.sum(y_true == y_pred)
    return hits / len(y_true) if normalize else hits


def confusion_matrix(y_true, y_pred, labels=None, normalize=None):
    """Rows are true labels, columns predicted. ``normalize`` in
    {None, 'true', 'pred', 'all'}."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = _labels(y_true, y_pred, labels)
    idx = {l: i for i, l in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=float)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    if normalize == "true":
        m = m / np.maximum(m.sum(1, keepdims=True), 1)
    elif normalize == "pred":
        m = m / np.maximum(m.sum(0, keepdims=True), 1)
    elif normalize == "all":
        m = m / m.sum()
    return m if normalize else m.astype(int)


def precision_recall_fscore_support(y_true, y_pred, beta=1.0, labels=None,
                                    pos_label=1, average=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labs = _labels(y_true, y_pred, labels)
    cm = confusion_matrix(y_true, y_pred, labs)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    support = cm.sum(1)

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    if average == "binary":
        if len(labs) > 2:
            raise ValueError("average='binary' needs a binary problem; use 'macro', "
                             "'micro' or 'weighted' for multiclass targets.")
        if pos_label not in labs:
            return 0.0, 0.0, 0.0, 0
        i = int(np.where(labs == pos_label)[0][0])
        p = div(tp[i], tp[i] + fp[i])
        r = div(tp[i], tp[i] + fn[i])
        f = div((1 + beta ** 2) * p * r, beta ** 2 * p + r)
        return float(p), float(r), float(f), int(support[i])
    if average == "micro":
        p = div(tp.sum(), tp.sum() + fp.sum())
        r = div(tp.sum(), tp.sum() + fn.sum())
        f = div((1 + beta ** 2) * p * r, beta ** 2 * p + r)
        return float(p), float(r), float(f), int(support.sum())
    p = div(tp, tp + fp)
    r = div(tp, tp + fn)
    f = div((1 + beta ** 2) * p * r, beta ** 2 * p + r)
    if average == "macro":
        return float(p.mean()), float(r.mean()), float(f.mean()), int(support.sum())
    if average == "weighted":
        w = support / support.sum()
        return float(p @ w), float(r @ w), float(f @ w), int(support.sum())
    return p, r, f, support


def precision_score(y_true, y_pred, pos_label=1, average="binary", labels=None):
    return precision_recall_fscore_support(y_true, y_pred, labels=labels, pos_label=pos_label, average=average)[0]


def recall_score(y_true, y_pred, pos_label=1, average="binary", labels=None):
    return precision_recall_fscore_support(y_true, y_pred, labels=labels, pos_label=pos_label, average=average)[1]


def f1_score(y_true, y_pred, pos_label=1, average="binary", labels=None):
    return precision_recall_fscore_support(y_true, y_pred, labels=labels, pos_label=pos_label, average=average)[2]


def fbeta_score(y_true, y_pred, beta, pos_label=1, average="binary", labels=None):
    return precision_recall_fscore_support(y_true, y_pred, beta=beta, labels=labels, pos_label=pos_label, average=average)[2]


def balanced_accuracy_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.diag(cm) / cm.sum(1)
    return float(np.nanmean(rec))


def cohen_kappa_score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return float((po - pe) / (1 - pe)) if pe != 1 else 1.0


def matthews_corrcoef(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).astype(float)
    t, p = cm.sum(1), cm.sum(0)
    c, s = np.trace(cm), cm.sum()
    cov_xy = c * s - t @ p
    cov_xx = s ** 2 - p @ p
    cov_yy = s ** 2 - t @ t
    denom = np.sqrt(cov_xx * cov_yy)
    return float(cov_xy / denom) if denom > 0 else 0.0


def roc_curve(y_true, y_score, pos_label=1):
    """Return ``fpr, tpr, thresholds`` (thresholds decreasing)."""
    y_true = (np.asarray(y_true) == pos_label).astype(float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_score, y_true = y_score[order], y_true[order]
    distinct = np.where(np.diff(y_score))[0]
    idx = np.r_[distinct, len(y_true) - 1]
    tps = np.cumsum(y_true)[idx]
    fps = 1 + idx - tps
    tpr = np.r_[0, tps / max(tps[-1], 1)]
    fpr = np.r_[0, fps / max(fps[-1], 1)]
    thresholds = np.r_[y_score[0] + 1, y_score[idx]]
    return fpr, tpr, thresholds


def auc(x, y):
    """Area under a curve by the trapezoidal rule."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, pos_label=1):
    """Binary AUC, or one-vs-rest macro AUC when ``y_score`` is an (n, k) matrix."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 2:
        if y_score.shape[1] == 2:
            y_score = y_score[:, 1]
        else:
            classes = np.unique(y_true)
            return float(np.mean([
                roc_auc_score(y_true == c, y_score[:, i], pos_label=True)
                for i, c in enumerate(classes)]))
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC AUC needs both classes present in y_true.")
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label)
    return auc(fpr, tpr)


def precision_recall_curve(y_true, y_score, pos_label=1):
    y_true = (np.asarray(y_true) == pos_label).astype(float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_score, y_true = y_score[order], y_true[order]
    distinct = np.where(np.diff(y_score))[0]
    idx = np.r_[distinct, len(y_true) - 1]
    tps = np.cumsum(y_true)[idx]
    fps = 1 + idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return np.r_[1, precision], np.r_[0, recall], y_score[idx]


def average_precision_score(y_true, y_score, pos_label=1):
    p, r, _ = precision_recall_curve(y_true, y_score, pos_label)
    return float(np.sum(np.diff(r) * p[1:]))


def log_loss(y_true, y_pred_proba, eps=1e-15, labels=None):
    """Cross-entropy. ``y_pred_proba`` may be 1-D (binary, P(class 1)) or (n, k)."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(y_pred_proba, dtype=float), eps, 1 - eps)
    if p.ndim == 1:
        classes = np.unique(y_true) if labels is None else np.asarray(labels)
        yb = (y_true == classes[-1]).astype(float) if len(classes) == 2 else y_true.astype(float)
        return float(-np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)))
    classes = np.unique(y_true) if labels is None else np.asarray(labels)
    idx = np.searchsorted(classes, y_true)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))


def brier_score_loss(y_true, y_prob, pos_label=1):
    y = (np.asarray(y_true) == pos_label).astype(float)
    return float(np.mean((np.asarray(y_prob, dtype=float) - y) ** 2))


def classification_report(y_true, y_pred, digits=2):
    """Text report with per-class precision / recall / F1 and averages."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred)
    labs = _labels(y_true, y_pred)
    w = max(12, max(len(str(l)) for l in labs) + 2)
    head = f"{'':>{w}}{'precision':>11}{'recall':>11}{'f1-score':>11}{'support':>10}\n\n"
    rows = "".join(f"{str(l):>{w}}{pi:>11.{digits}f}{ri:>11.{digits}f}{fi:>11.{digits}f}{si:>10d}\n"
                   for l, pi, ri, fi, si in zip(labs, p, r, f, s))
    acc = accuracy_score(y_true, y_pred)
    n = int(s.sum())
    out = head + rows + f"\n{'accuracy':>{w}}{'':>11}{'':>11}{acc:>11.{digits}f}{n:>10d}\n"
    for avg in ("macro", "weighted"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, average=avg)
        out += f"{avg + ' avg':>{w}}{pa:>11.{digits}f}{ra:>11.{digits}f}{fa:>11.{digits}f}{n:>10d}\n"
    return out
