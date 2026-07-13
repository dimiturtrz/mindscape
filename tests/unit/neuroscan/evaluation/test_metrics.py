"""Equivalence-class unit tests for the pure metrics."""
import numpy as np

from neuroscan.evaluation import metrics


def test_accuracy_perfect_and_zero():
    y = np.array([0, 1, 2, 3])
    assert metrics.Metrics.accuracy(y, y) == 1.0
    assert metrics.Metrics.accuracy(y, (y + 1) % 4) == 0.0


def test_kappa_perfect_is_one():
    y = np.array([0, 1, 0, 1, 2, 2])
    assert metrics.Metrics.kappa(y, y) == 1.0


def test_ece_perfectly_calibrated_is_zero():
    # confidence == accuracy in every bin -> ECE 0
    conf = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([1.0, 1.0, 1.0, 1.0])
    e, _ = metrics.Metrics.ece(conf, correct)
    assert e == 0.0


def test_ece_maximally_miscalibrated():
    # fully confident, always wrong -> ECE 1
    conf = np.array([1.0, 1.0])
    correct = np.array([0.0, 0.0])
    e, _ = metrics.Metrics.ece(conf, correct)
    assert abs(e - 1.0) < 1e-9


def test_ece_from_probs_matches_argmax():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    y = np.array([0, 1])           # both correct, conf 0.9/0.8
    assert metrics.Metrics.ece_from_probs(probs, y) >= 0.0


def test_confusion_shape_and_counts():
    y = np.array([0, 0, 1, 2])
    p = np.array([0, 1, 1, 2])
    cm = metrics.Metrics.confusion(y, p, 3)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1 and cm[0, 1] == 1 and cm[1, 1] == 1 and cm[2, 2] == 1
    assert cm.sum() == 4


def test_boot_ci_brackets_known_rate():
    """A hit vector with a known 20% rate -> point is exact, and the percentile CI brackets it."""
    hits = np.zeros(1000)
    hits[:200] = 1.0                                          # 20% hit rate
    point, lo, hi = metrics.Metrics.boot_ci(np.mean, hits)
    assert point == 0.2
    assert lo < 0.2 < hi
    assert hi - lo < 0.1                                      # N=1000 -> a tight interval around 20%


def test_boot_ci_degenerate_all_hit_is_point_mass():
    """All-ones (or all-zeros) -> every resample is the same mean -> a zero-width CI at the point."""
    point, lo, hi = metrics.Metrics.boot_ci(np.mean, np.ones(50))
    assert point == 1.0 and lo == 1.0 and hi == 1.0


def test_boot_delta_ci_identical_arms_straddle_zero():
    """Paired delta of an arm against ITSELF -> delta exactly 0 and the CI is a point mass at 0 (the same
    resampled indices hit both arms, so every draw cancels)."""
    rng = np.random.default_rng(1)
    hits = (rng.random(500) < 0.3).astype(float)
    delta, lo, hi = metrics.Metrics.boot_delta_ci(np.mean, [hits], [hits])
    assert delta == 0.0 and lo == 0.0 and hi == 0.0


def test_boot_delta_ci_separated_arms_exclude_zero():
    """A clear per-trial gap (arm B hits everywhere A misses and more) -> the paired delta CI excludes 0."""
    a = np.zeros(400)
    a[:40] = 1.0                                              # 10%
    b = np.zeros(400)
    b[:240] = 1.0                                             # 60%, superset of A's hits
    delta, lo, hi = metrics.Metrics.boot_delta_ci(np.mean, [a], [b])
    assert abs(delta - 0.5) < 1e-9
    assert lo > 0.0                                           # honest 'B beats A' — CI clears 0
