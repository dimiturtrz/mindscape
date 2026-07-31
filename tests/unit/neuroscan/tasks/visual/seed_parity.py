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


def test_run(monkeypatch):
    """run collates per-seed results from both arms and computes the naive-minus-optimized gap."""
    # stub TrainNice.train to return synthetic result dicts
    def stub_train(train_subj, test_subj, cfg):
        return {
            "single_trial": {"1": 0.02, "5": 0.09},
            "concept_avg": {"1": 0.04, "5": 0.15},
        }
    from neuroscan.tasks.visual import train_nice
    monkeypatch.setattr(train_nice.TrainNice, "train", stub_train)

    result = SeedParity.run(train_subjects=[1, 2], test_subject=3, seeds=[0, 1])
    assert result["train"] == [1, 2]
    assert result["test"] == 3
    assert result["seeds"] == [0, 1]
    assert "arms" in result
    assert "naive" in result["arms"] and "optimized" in result["arms"]
    assert "gap_naive_minus_optimized" in result
    assert "single_trial_top1" in result["gap_naive_minus_optimized"]
