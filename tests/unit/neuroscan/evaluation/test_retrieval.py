"""Retrieval confidence calibration — pure, on synthetic candidate-score matrices."""
import numpy as np

from neuroscan.evaluation.retrieval import retrieval_calibration


def _scores(n_conf_correct, n_flat_wrong, n_cand=5, seed=0):
    """Build [N, C] scores + labels: `n_conf_correct` trials with a sharp peak ON the true label (confident +
    correct), `n_flat_wrong` with the true label pushed slightly BELOW the rest (low confidence + wrong)."""
    rng = np.random.default_rng(seed)
    scores, labels = [], []
    for _ in range(n_conf_correct):
        y = int(rng.integers(n_cand))
        row = rng.normal(0, 0.01, n_cand)
        row[y] += 8.0                                   # sharp peak on the true label
        scores.append(row); labels.append(y)
    for _ in range(n_flat_wrong):
        y = int(rng.integers(n_cand))
        row = rng.normal(0, 0.01, n_cand)
        row[y] -= 0.5                                   # true label below the flat rest -> argmax elsewhere
        scores.append(row); labels.append(y)
    return np.array(scores), np.array(labels)


def test_informative_confidence_positive_gap_and_counts():
    scores, labels = _scores(60, 40)
    out = retrieval_calibration(scores, labels, scale=1.0, n_bins=10)
    assert abs(out["top1_acc"] - 0.60) < 0.05           # the 60 sharp trials are the hits
    assert out["conf_gap"] > 0.1                         # confident trials are the correct ones
    assert 0.0 <= out["ece"] <= 1.0
    assert sum(b["count"] for b in out["reliability"]) == 100   # every trial lands in exactly one bin


def test_all_correct_gap_is_nan():
    scores, labels = _scores(30, 0)
    out = retrieval_calibration(scores, labels)
    assert out["top1_acc"] == 1.0
    assert np.isnan(out["conf_gap"])                     # no misses -> gap undefined, reported as nan


def test_scale_sharpens_confidence():
    # higher temperature scale concentrates softmax -> higher mean confidence on the same scores
    scores, labels = _scores(50, 50, seed=3)
    lo = retrieval_calibration(scores, labels, scale=0.5)
    hi = retrieval_calibration(scores, labels, scale=5.0)
    lo_conf = sum(b["conf"] * b["count"] for b in lo["reliability"]) / 100
    hi_conf = sum(b["conf"] * b["count"] for b in hi["reliability"]) / 100
    assert hi_conf > lo_conf
