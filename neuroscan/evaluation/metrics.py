"""Pure decoding metrics — accuracy, Cohen's kappa, calibration (ECE/Brier), confusion.

All take numpy arrays, no IO, no side effects (unit-testable). `ece` is the calibration headline:
the gap between confidence and accuracy that domain shift blows open (the siblings' signature, carried).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score


@dataclass(frozen=True)
class BootCfg:
    """Bootstrap knobs: resample count, two-sided level, and the resampling RNG (a bootstrap seed,
    unrelated to any training seed). `rng=None` -> a fixed default_rng(0) so intervals are reproducible."""

    n_boot: int = 1000
    alpha: float = 0.05
    rng: object = None

    def gen(self):
        return np.random.default_rng(0) if self.rng is None else self.rng


_DEFAULT_BOOT = BootCfg()   # frozen -> a safe shared default (no per-call mutation)


class Metrics:
    """Pure decoding metrics + a test-set bootstrap for honest uncertainty from ONE run (no retrains).

    `boot_ci` resamples the N test trials with replacement, recomputes the metric per draw, and reports the
    percentile interval — power comes from the test-set N, so one run yields `point [lo, hi]`. `boot_delta_ci`
    is the PAIRED A-vs-B version: the SAME resampled trial indices hit both runs each draw, so shared test-set
    noise cancels and the interval is on the delta (does B beat A honestly?). For retrieval the metric is just
    `np.mean` over a per-trial hit vector (`Nice.retrieval_hits`); the fold-free answer to bd 5s3l/s1t2."""

    @staticmethod
    def _resample(arrays, idx):
        """Index the first axis (= trial) of every metric-input array by one bootstrap draw."""
        return [np.asarray(a)[idx] for a in arrays]

    @staticmethod
    def _interval(point, samples, alpha):
        """(point, lo, hi) — percentile interval over the finite bootstrap samples (NaNs dropped)."""
        s = np.asarray(samples, dtype=np.float64)
        s = s[np.isfinite(s)]
        if s.size == 0:
            return (point, float("nan"), float("nan"))
        lo, hi = np.percentile(s, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return (point, float(lo), float(hi))

    @staticmethod
    def boot_ci(metric_fn, *arrays, cfg: BootCfg = _DEFAULT_BOOT) -> tuple[float, float, float]:
        """(point, lo, hi) for `metric_fn(*arrays)` — percentile bootstrap over the test trials. `metric_fn`
        takes the arrays in order (retrieval: `np.mean` over one hit vector); every array's first axis is the
        trial, resampled by a shared index."""
        arrays = [np.asarray(a) for a in arrays]
        n = len(arrays[0])
        point = float(metric_fn(*arrays))
        draws = cfg.gen().integers(0, n, size=(cfg.n_boot, n))
        samples = [metric_fn(*Metrics._resample(arrays, idx)) for idx in draws]
        return Metrics._interval(point, samples, cfg.alpha)

    @staticmethod
    def boot_delta_ci(metric_fn, arrays_a, arrays_b, cfg: BootCfg = _DEFAULT_BOOT) -> tuple[float, float, float]:
        """(delta, lo, hi) for metric(B) − metric(A), PAIRED bootstrap. Same resampled trial indices hit both
        runs each draw, so shared test-set noise cancels — a CI that excludes 0 is an honest 'B differs from A'.
        Requires A and B scored on the SAME test trials in the SAME order."""
        arrays_a = [np.asarray(a) for a in arrays_a]
        arrays_b = [np.asarray(a) for a in arrays_b]
        n = len(arrays_a[0])
        delta = float(metric_fn(*arrays_b)) - float(metric_fn(*arrays_a))
        draws = cfg.gen().integers(0, n, size=(cfg.n_boot, n))
        samples = [metric_fn(*Metrics._resample(arrays_b, idx)) - metric_fn(*Metrics._resample(arrays_a, idx))
                   for idx in draws]
        return Metrics._interval(delta, samples, cfg.alpha)

    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float((np.asarray(y_true) == np.asarray(y_pred)).mean())

    @staticmethod
    def kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Cohen's kappa — chance-corrected agreement; the standard BCI motor-imagery metric."""
        return float(cohen_kappa_score(y_true, y_pred))

    @staticmethod
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

    @staticmethod
    def ece_from_probs(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
        """ECE straight from a [n, C] probability matrix + true labels."""
        probs = np.asarray(probs, float)
        pred, conf = probs.argmax(1), probs.max(1)
        return Metrics.ece(conf, (pred == np.asarray(y_true)).astype(float), n_bins)[0]

    @staticmethod
    def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
        """[n_classes, n_classes] integer confusion matrix (rows = true, cols = pred)."""
        t, p = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
        flat = np.bincount(t * n_classes + p, minlength=n_classes * n_classes)   # 2D histogram, vectorized
        return flat.reshape(n_classes, n_classes)
