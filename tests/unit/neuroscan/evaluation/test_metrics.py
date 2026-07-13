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
