"""Stratified diagnostics (`diagnostics.by_group` / `diagnostics.spread`) — break an aggregate metric down
by subject/session. Pure: synthetic labels + preds, known per-group accuracies."""
import numpy as np

from neuroscan.evaluation import diagnostics


def test_spread_summarizes_across_groups():
    rows = [{"acc": 0.4}, {"acc": 0.6}, {"acc": 0.8}]
    s = diagnostics.Diagnostics.spread(rows, key="acc")
    assert np.isclose(s["mean"], 0.6)
    assert np.isclose(s["min"], 0.4) and np.isclose(s["max"], 0.8)
    assert np.isclose(s["std"], np.std([0.4, 0.6, 0.8]))
