"""Equivalence-class tests for the paired bootstrap delta between two retrieval runs (bd s1t2)."""
import numpy as np
import pytest

from neuroscan.evaluation.metrics import BootCfg
from neuroscan.tasks.visual.paired_delta import PairedDelta

_CFG = BootCfg(n_boot=500)


def test_separated_arms_delta_excludes_zero():
    """B hits a clear superset of A's trials -> the paired delta CI clears 0 (an honest 'B beats A')."""
    a = {1: np.array([1.0] * 20 + [0.0] * 180)}          # 10%
    b = {1: np.array([1.0] * 120 + [0.0] * 80)}          # 60%, superset
    out = PairedDelta.compare(a, b, _CFG)
    assert abs(out[1]["a"][0] - 0.10) < 1e-9 and abs(out[1]["b"][0] - 0.60) < 1e-9
    assert out[1]["delta"][1] > 0                          # lo > 0


def test_identical_arms_delta_is_point_mass_at_zero():
    """A vs itself -> paired resampling cancels every draw -> delta CI is a point mass at 0."""
    hits = {1: (np.arange(300) % 5 == 0).astype(float)}
    out = PairedDelta.compare(hits, hits, _CFG)
    assert out[1]["delta"] == (0.0, 0.0, 0.0)


def test_mismatched_lengths_raise():
    a = {1: np.ones(100)}
    b = {1: np.ones(90)}                                   # not paired
    with pytest.raises(ValueError, match="not paired"):
        PairedDelta.compare(a, b, _CFG)


def test_hits_reads_string_keyed_json_result():
    """`train --out` writes JSON, so single_trial_hits keys are strings — _hits must still read k=1."""
    result = {"single_trial_hits": {"1": [1.0, 0.0, 1.0], "5": [1.0, 1.0, 1.0]}}
    assert PairedDelta._hits(result, 1).tolist() == [1.0, 0.0, 1.0]
