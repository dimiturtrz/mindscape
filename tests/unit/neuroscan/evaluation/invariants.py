"""Equivalence-class tests for the run-end retrieval invariants (bd qpdr)."""
import numpy as np
import pytest

from neuroscan.evaluation.invariants import Invariants
from neuroscan.evaluation.metrics import Metrics


def _clean():
    """A self-consistent retrieval result: CI points equal the means, brackets contain them, above chance."""
    return {"single_trial": {1: 0.02, 5: 0.09},
            "single_trial_ci": {1: (0.02, 0.015, 0.025), 5: (0.09, 0.08, 0.10)},
            "chance_top1": 0.005}


def test_check():
    # clean result has no violations
    assert Invariants.check(_clean()) == []
    # ci point-mean mismatch flagged
    res = _clean()
    res["single_trial_ci"][1] = (0.03, 0.02, 0.04)      # point 0.03 != reported mean 0.02
    v = Invariants.check(res)
    assert len(v) == 1 and "aggregate mismatch" in v[0]
    # bracket excluding point flagged
    res = _clean()
    res["single_trial_ci"][5] = (0.09, 0.10, 0.11)      # point 0.09 below the lo 0.10
    assert any("excludes point" in m for m in Invariants.check(res))
    # below chance top1 flagged
    res = _clean()
    res["single_trial"][1] = 0.002                       # under chance 0.005
    res["single_trial_ci"][1] = (0.002, 0.001, 0.004)
    assert any("below chance" in m for m in Invariants.check(res))
    # out of range accuracy flagged
    res = _clean()
    res["single_trial"][1] = 1.5                         # not a probability
    res["single_trial_ci"][1] = (1.5, 1.4, 1.6)
    assert any("not a finite [0,1]" in m for m in Invariants.check(res))
    # nan metric is skipped not crashed
    res = _clean()
    res["single_trial"][1] = float("nan")                # undefined -> skip its consistency checks, don't raise
    res["single_trial_ci"][1] = (float("nan"), float("nan"), float("nan"))
    Invariants.check(res)                                # no exception
    # strict raises on violation
    res = _clean()
    res["single_trial_ci"][1] = (0.03, 0.02, 0.04)
    with pytest.raises(AssertionError):
        Invariants.check(res, strict=True)
    # string keys after json roundtrip accepted
    res = {"single_trial": {"1": 0.02, "5": 0.09},
           "single_trial_ci": {"1": [0.02, 0.015, 0.025], "5": [0.09, 0.08, 0.10]},
           "chance_top1": 0.005}
    assert Invariants.check(res) == []


def test_reconciles():
    assert Invariants.reconciles(0.007, 0.016, 0.023)    # 0.023 - 0.016 == 0.007
    assert not Invariants.reconciles(0.010, 0.016, 0.023)
    # matches boot_delta: delta from boot_delta_ci equals point_b − point_a
    a = (np.arange(100) < 16).astype(float)
    b = (np.arange(100) < 23).astype(float)
    delta, _lo, _hi = Metrics.boot_delta_ci(np.mean, [a], [b])
    assert Invariants.reconciles(delta, float(a.mean()), float(b.mean()))
