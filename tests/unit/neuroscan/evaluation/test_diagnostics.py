"""Stratified diagnostics (`diagnostics.by_group` / `diagnostics.spread`) — break an aggregate metric down
by subject/session. Pure: synthetic labels + preds, known per-group accuracies."""
import numpy as np

from neuroscan.evaluation import diagnostics


def test_by_group_computes_per_group_rows_sorted():
    group = np.array(["s2", "s2", "s1", "s1"])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])            # s2 all-correct (1.0), s1 one wrong (0.5)
    rows = diagnostics.by_group(group, y_true, y_pred)
    assert [r["group"] for r in rows] == ["s1", "s2"]       # sorted by group
    assert rows[0]["n"] == 2 and rows[1]["n"] == 2
    assert rows[0]["acc"] == 0.5 and rows[1]["acc"] == 1.0
    assert "ece" not in rows[0]                             # no probs -> no ece key


def test_by_group_includes_ece_when_probs_given():
    group = np.array([0, 0, 1, 1])
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    probs = np.array([[.9, .1], [.2, .8], [.7, .3], [.4, .6]])
    rows = diagnostics.by_group(group, y_true, y_pred, probs)
    assert all("ece" in r for r in rows)
    assert all(0.0 <= r["ece"] <= 1.0 for r in rows)


def test_spread_summarizes_across_groups():
    rows = [{"acc": 0.4}, {"acc": 0.6}, {"acc": 0.8}]
    s = diagnostics.spread(rows, key="acc")
    assert np.isclose(s["mean"], 0.6)
    assert np.isclose(s["min"], 0.4) and np.isclose(s["max"], 0.8)
    assert np.isclose(s["std"], np.std([0.4, 0.6, 0.8]))
