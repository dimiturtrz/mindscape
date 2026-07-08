"""Calibration diagnostics for zero-shot retrieval — does the model KNOW when it's right?

Cross-subject EEG->image retrieval reports a low top-k (the honest number). A deployable system also needs
the confidence to be trustworthy: when the retrieval is confident, is it actually right? This module turns
the per-trial candidate scores into a confidence (softmax over the candidate bank) and measures how well that
confidence tracks correctness — the expected calibration error (ECE), a reliability curve, and the gap between
the mean confidence on hits vs misses. Pure numpy so it's testable without a trained encoder.
"""
from __future__ import annotations

import numpy as np


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def retrieval_calibration(scores: np.ndarray, labels: np.ndarray, *, scale: float = 1.0,
                          n_bins: int = 10) -> dict:
    """Confidence calibration of a retrieval head. `scores` [N, C] = per-trial similarity to each of C
    candidates (e.g. cosine); `labels` [N] = the true candidate index. Confidence = softmax(scale * scores)
    at the predicted candidate. Returns top-1 accuracy, ECE (equal-width confidence bins), the reliability
    curve (per-bin mean-confidence, accuracy, count), and `conf_gap` = mean confidence on hits minus on misses
    (positive => confidence is informative). `scale` mimics the trained logit temperature (raw cosine is flat).
    """
    probs = _softmax(scale * np.asarray(scores, dtype=float))
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = predicted == np.asarray(labels)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if not count:
            continue
        bin_conf = float(confidence[in_bin].mean())
        bin_acc = float(correct[in_bin].mean())
        ece += (count / len(confidence)) * abs(bin_acc - bin_conf)
        bins.append({"conf": bin_conf, "acc": bin_acc, "count": count})

    hits, misses = confidence[correct], confidence[~correct]
    conf_gap = float(hits.mean() - misses.mean()) if len(hits) and len(misses) else float("nan")
    return {"top1_acc": float(correct.mean()), "ece": float(ece), "conf_gap": conf_gap, "reliability": bins}
