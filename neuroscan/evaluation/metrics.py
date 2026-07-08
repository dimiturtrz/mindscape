"""Pure decoding metrics — accuracy, Cohen's kappa, calibration (ECE/Brier), confusion.

All take numpy arrays, no IO, no side effects (unit-testable). `ece` is the calibration headline:
the gap between confidence and accuracy that domain shift blows open (the siblings' signature, carried).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa — chance-corrected agreement; the standard BCI motor-imagery metric."""
    return float(cohen_kappa_score(y_true, y_pred))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, list]:
    """Expected Calibration Error + per-bin (conf, acc, weight) for a reliability diagram.
    `conf` = max-softmax confidence per sample; `correct` = 1.0/0.0 whether the argmax was right."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    e, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        c, a, w = conf[m].mean(), correct[m].mean(), m.mean()
        e += w * abs(a - c)
        bins.append((float(c), float(a), float(w)))
    return float(e), bins


def ece_from_probs(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE straight from a [n, C] probability matrix + true labels."""
    probs = np.asarray(probs, float)
    pred, conf = probs.argmax(1), probs.max(1)
    return ece(conf, (pred == np.asarray(y_true)).astype(float), n_bins)[0]


def brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Multiclass Brier score = mean squared error between probs and the one-hot truth."""
    probs = np.asarray(probs, float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), np.asarray(y_true)] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """[n_classes, n_classes] integer confusion matrix (rows = true, cols = pred)."""
    t, p = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    flat = np.bincount(t * n_classes + p, minlength=n_classes * n_classes)   # 2D histogram, vectorized
    return flat.reshape(n_classes, n_classes)
