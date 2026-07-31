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


def test_clean_result_has_no_violations():
    assert Invariants.check(_clean()) == []


def test_ci_point_mean_mismatch_flagged():
    res = _clean()
    res["single_trial_ci"][1] = (0.03, 0.02, 0.04)      # point 0.03 != reported mean 0.02
    v = Invariants.check(res)
    assert len(v) == 1 and "aggregate mismatch" in v[0]


def test_bracket_excluding_point_flagged():
    res = _clean()
    res["single_trial_ci"][5] = (0.09, 0.10, 0.11)      # point 0.09 below the lo 0.10
    assert any("excludes point" in m for m in Invariants.check(res))


def test_below_chance_top1_flagged():
    res = _clean()
    res["single_trial"][1] = 0.002                       # under chance 0.005
    res["single_trial_ci"][1] = (0.002, 0.001, 0.004)
    assert any("below chance" in m for m in Invariants.check(res))


def test_out_of_range_accuracy_flagged():
    res = _clean()
    res["single_trial"][1] = 1.5                         # not a probability
    res["single_trial_ci"][1] = (1.5, 1.4, 1.6)
    assert any("not a finite [0,1]" in m for m in Invariants.check(res))


def test_nan_metric_is_skipped_not_crashed():
    res = _clean()
    res["single_trial"][1] = float("nan")                # undefined -> skip its consistency checks, don't raise
    res["single_trial_ci"][1] = (float("nan"), float("nan"), float("nan"))
    Invariants.check(res)                                # no exception


def test_strict_raises_on_violation():
    res = _clean()
    res["single_trial_ci"][1] = (0.03, 0.02, 0.04)
    with pytest.raises(AssertionError):
        Invariants.check(res, strict=True)


def test_string_keys_after_json_roundtrip_accepted():
    res = {"single_trial": {"1": 0.02, "5": 0.09},
           "single_trial_ci": {"1": [0.02, 0.015, 0.025], "5": [0.09, 0.08, 0.10]},
           "chance_top1": 0.005}
    assert Invariants.check(res) == []


def test_reconciles_paired_delta():
    assert Invariants.reconciles(0.007, 0.016, 0.023)    # 0.023 - 0.016 == 0.007
    assert not Invariants.reconciles(0.010, 0.016, 0.023)


def test_reconciles_matches_boot_delta():
    """The s1t2 guard on real bootstrap output: delta from boot_delta_ci equals point_b − point_a."""
    a = (np.arange(100) < 16).astype(float)
    b = (np.arange(100) < 23).astype(float)
    delta, _lo, _hi = Metrics.boot_delta_ci(np.mean, [a], [b])
    assert Invariants.reconciles(delta, float(a.mean()), float(b.mean()))
