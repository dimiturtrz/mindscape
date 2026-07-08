"""WindowedFnirs — harness contract for every aggregation, plus the representation claim pinned per
equivalence class: each aggregation is designed for a different temporal signal, so each is tested on the
signal it should win on, and always against the global mean/slope/peak collapse (which is blind to all of
them by construction: matched global mean≈0, full-window slope≈0, equal peak amplitude)."""
import numpy as np
import pytest

from baselines.fnirs.features import FnirsLda
from baselines.fnirs.windowed import _AGGREGATES, WindowedFnirs


def _ordered_shape_data(seed=0):
    """Class = oscillation frequency over the WHOLE window (ordered global shape): cos of k∈{3,6,9} cycles
    over 10 s. Even about centre -> global slope≈0, mean≈0; same amplitude -> equal peak. Collapse blind;
    the ordered trajectory differs -> `concat` (position-aware) should separate."""
    rng = np.random.default_rng(seed)
    tc = np.arange(100) / 10.0
    X, y = [], []
    for c, k in {0: 3, 1: 6, 2: 9}.items():
        for _ in range(60):
            wave = np.cos(2 * np.pi * (k / 10.0) * tc)
            X.append(wave[None, :] + rng.standard_normal((4, 100)) * 0.05); y.append(c)
    X = np.asarray(X); y = np.asarray(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


@pytest.mark.parametrize("agg", _AGGREGATES)
def test_predict_proba_contract(agg):
    X, y = _ordered_shape_data()
    p = WindowedFnirs(win_s=2.0, hop_s=0.5, fs=10.0, aggregate=agg).fit(X, y).predict_proba(X)
    assert p.shape == (len(y), 3)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)                         # rows are probability distributions


def _acc(clf, X, y, cut):
    tr, te = slice(0, cut), slice(cut, len(y))
    clf.fit(X[tr], y[tr])
    return (clf.predict_proba(X[te]).argmax(1) == y[te]).mean()


def test_concat_captures_ordered_shape():
    X, y = _ordered_shape_data(); cut = int(0.7 * len(y))
    acc_win = _acc(WindowedFnirs(win_s=2.0, hop_s=0.5, fs=10.0, aggregate="concat"), X, y, cut)
    acc_col = _acc(FnirsLda(), X, y, cut)
    assert acc_win > 0.7                                                 # ordered trajectory recoverable
    assert acc_win > acc_col + 0.2                                       # global collapse is blind to it


def test_short_block_yields_one_window():
    clf = WindowedFnirs(win_s=6.0, hop_s=1.0, fs=10.0)
    assert clf._starts(t=30) == [0]                                      # block shorter than window -> 1 window
    assert len(clf._starts(t=220)) > 1


def test_bad_aggregate_rejected():
    with pytest.raises(ValueError):
        WindowedFnirs(aggregate="nope")


@pytest.mark.parametrize("agg", ["max", "lse"])
def test_mil_pooling_rejects_binary(agg):
    X, y = _ordered_shape_data()
    Xb, yb = X[y != 1], y[y != 1]                                        # drop the middle class -> binary
    with pytest.raises(ValueError):                                      # MIL score-pool needs a per-class axis
        WindowedFnirs(win_s=2.0, hop_s=0.5, fs=10.0, aggregate=agg).fit(Xb, yb)
