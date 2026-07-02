"""Fusion combiners/diagnostics — pure functions on per-block probabilities / correct-masks. Synthetic."""
import numpy as np

from baselines.fusion import combine


def test_complementarity_oracle_and_error_structure():
    #        block: 0    1    2    3
    ce = np.array([1, 1, 0, 0], bool)      # EEG correct on 0,1
    cf = np.array([1, 0, 1, 0], bool)      # fNIRS correct on 0,2 — they disagree on 1,2
    comp = combine.complementarity({"eeg": 0.5, "fnirs": 0.5}, ce, cf)
    assert comp["best_single"] == 0.5
    assert comp["oracle_either"] == 0.75           # either right on 0,1,2 -> 3/4
    assert comp["both_correct"] == 0.25 and comp["both_wrong"] == 0.25
    assert comp["eeg_only"] == 0.25 and comp["fnirs_only"] == 0.25


def test_aggregation_sweep_mean_and_confgap():
    y = np.array([0, 1, 0, 1])
    # a confident-and-right EEG on some blocks, fNIRS elsewhere
    Pe = np.array([[.9, .1], [.1, .9], [.4, .6], [.6, .4]])   # right on 0,1
    Pf = np.array([[.6, .4], [.6, .4], [.8, .2], [.2, .8]])   # right on 0,2,3
    ce = Pe.argmax(1) == y
    cf = Pf.argmax(1) == y
    agg = combine.aggregation_sweep(Pe, Pf, Pe, Pe, Pf, y, ce, cf)
    assert set(combine.SWEEP_KEYS) <= set(agg)                # every sweep key present
    assert 0.0 <= agg["mean"] <= 1.0
    # conf_gap = mean max-prob(correct) - mean max-prob(wrong); finite, sign as computed
    assert np.isfinite(agg["eeg_conf_gap"]) and np.isfinite(agg["fnirs_conf_gap"])
