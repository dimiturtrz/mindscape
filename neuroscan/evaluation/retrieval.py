"""Retrieval quality + calibration diagnostics for zero-shot EEG->image.

`retrieval_metrics` — richer than top-k: recall@k, MRR, median rank, PR-AUC (top-1 alone hides whether the
true concept landed at rank 2 or rank 200). `retrieval_calibration` — does the model KNOW when it's right?:
softmax the candidate scores into a confidence, then ECE + a reliability curve + the hit-vs-miss confidence
gap. Both pure (numpy/sklearn on the [N, C] score matrix), testable without a trained encoder.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def retrieval_metrics(scores: np.ndarray, labels: np.ndarray, ks: tuple[int, ...] = (1, 5)) -> dict:
    """Full retrieval quality from the [N, C] candidate-score matrix + true label per trial — richer than
    top-k, which discards WHERE the true concept ranked (rank 2 vs rank C look identical at top-1).

    Returns recall@k, MRR (mean reciprocal rank), median rank, and PR-AUC. Rank is 1-based (# candidates
    scoring strictly higher than the true one, +1). Note: mean-average-precision equals MRR here because each
    trial has exactly one relevant candidate, so it isn't reported separately. PR-AUC (over all N×C pairs,
    positive = the true candidate) is the unbiased AUC under the 1-vs-(C-1) imbalance — ROC-AUC inflates.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    n = len(labels)
    true_score = scores[np.arange(n), labels]
    ranks = (scores > true_score[:, None]).sum(axis=1) + 1        # 1-based rank of the true candidate
    out = {f"recall@{k}": float((ranks <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / ranks).mean())
    out["median_rank"] = float(np.median(ranks))
    relevant = np.zeros_like(scores, dtype=int)
    relevant[np.arange(n), labels] = 1
    out["pr_auc"] = float(average_precision_score(relevant.ravel(), scores.ravel()))
    return out


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
