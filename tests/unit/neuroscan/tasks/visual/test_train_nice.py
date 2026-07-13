"""NICE trainer pure helpers — the DANN lambda schedule (no training, no data).

`TrainNice._dann_lambda` ramps the gradient-reversal strength 0 -> max over training progress. Equivalence classes:
start (0), end (~max), monotonic increase, and scaling by max_lambda.
"""
from neuroscan.tasks.visual.train_nice import TrainNice


def test_dann_lambda_starts_at_zero():
    assert TrainNice._dann_lambda(0.0, 1.0) == 0.0                       # random early encoder -> adversary off


def test_dann_lambda_approaches_max_at_end():
    assert 0.9 < TrainNice._dann_lambda(1.0, 1.0) <= 1.0                 # near full strength by the end


def test_dann_lambda_monotonic_and_scales():
    vals = [TrainNice._dann_lambda(p / 10, 2.0) for p in range(11)]
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False))   # non-decreasing over progress
    assert vals[-1] <= 2.0 and vals[0] == 0.0                  # scaled by max_lambda=2
