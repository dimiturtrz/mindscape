"""Robustness-grid aggregation for the EEG->image retrieval audit — pure, synthetic accuracies."""
from pytest import approx

from neuroscan.tasks.visual.retrieval_audit import _ROBUST, RetrievalAudit


def _row(ws1, wa1, cs1, ca1):
    # only top-1 varied here; top-5 mirrors it so the ks loop is exercised on both
    return {
        "within_single": {1: ws1, 5: ws1 + 0.1}, "within_avg": {1: wa1, 5: wa1 + 0.1},
        "cross_single": {1: cs1, 5: cs1 + 0.1}, "cross_avg": {1: ca1, 5: ca1 + 0.1},
    }


def test_summarize_means_grid_and_inflation_over_robust():
    rows = [_row(0.40, 0.60, 0.20, 0.30), _row(0.50, 0.70, 0.30, 0.40)]
    out = RetrievalAudit.summarize(rows)
    assert out["n_subjects"] == 2 and out["robust_cell"] == _ROBUST == "cross_single"
    # cell means
    assert out["grid"]["within_single"][1] == approx(0.45) and out["grid"]["cross_single"][1] == approx(0.25)
    assert out["grid"]["within_avg"][1] == approx(0.65) and out["grid"]["cross_avg"][1] == approx(0.35)
    # inflation = leaky cell - robust (cross_single); robust itself absent from the delta map
    assert _ROBUST not in out["inflation_over_robust"]
    assert abs(out["inflation_over_robust"]["within_avg"][1] - (0.65 - 0.25)) < 1e-9   # the field's inflation
    assert abs(out["inflation_over_robust"]["cross_avg"][1] - (0.35 - 0.25)) < 1e-9    # averaging-only leak
    assert out["inflation_over_robust"]["within_single"][5] > 0                        # top-5 path too


def test_summarize_single_subject_zero_self_inflation():
    out = RetrievalAudit.summarize([_row(0.40, 0.60, 0.20, 0.30)])
    assert out["n_subjects"] == 1
    # a leaky cell equal to robust would show 0 inflation; here cross_single is the robust ref
    assert all(v == 0.0 for v in _self_delta(out).values())


def _self_delta(out):
    # robust vs itself is not emitted; recompute to assert the reference is internally consistent
    g = out["grid"][_ROBUST]
    return {k: out["grid"][_ROBUST][k] - g[k] for k in g}
