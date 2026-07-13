"""Seed-parity aggregation (`_agg`) — mean/std across seeds for one metric/k, no training.

Equivalence classes: single seed (std must be 0, not a crash), multiple seeds (population std),
and the value list is carried through for auditability.
"""
import math

from neuroscan.tasks.visual.seed_parity import SeedParity

_RUNS = [
    {"single_trial": {"1": 0.02, "5": 0.09}, "concept_avg": {"1": 0.04, "5": 0.15}},
    {"single_trial": {"1": 0.04, "5": 0.11}, "concept_avg": {"1": 0.06, "5": 0.17}},
]


def test_agg_multi_seed_mean_std_and_vals():
    a = SeedParity._agg(_RUNS, "single_trial", "1")
    assert a["mean"] == 0.03                                   # (0.02 + 0.04) / 2
    assert math.isclose(a["std"], 0.01)                        # population std of {0.02, 0.04}
    assert a["vals"] == [0.02, 0.04]                           # raw draws kept


def test_agg_single_seed_zero_std():
    a = SeedParity._agg(_RUNS[:1], "concept_avg", "5")
    assert a["mean"] == 0.15 and a["std"] == 0.0               # one draw -> no spread, not a divide error
