"""Equivalence-class tests for the README numbers-sync (results.py + sync_numbers.py).

Pure logic only — schema normalization, run-name parsing, marker rendering — no runs/ tree, no real README.
The representative metric values live in the constants below so assertions read as intent, not scattered
literals; the sample acc is deliberately unrounded to exercise snapshot precision + display formatting.
"""
import json

import pytest

from neuroscan.evaluation import results, sync_numbers

ACC, KAPPA, ECE = 0.59761, 0.4635, 0.1395            # one representative run's metrics
CROSS_ACC = 0.3823                                    # a second run, for the within→cross gap
NAME = "csp_lda_within_bnci2014_001"


def _harness(acc=ACC, kappa=KAPPA, ece=ECE):
    return {"n_classes": 4, "fold_mean": {"acc": acc, "kappa": kappa, "ece": ece}}


def _write_run(tmp_path, name, agg=None):
    d = tmp_path / name
    d.mkdir()
    (d / "aggregate.json").write_text(json.dumps(agg or _harness()))
    return d


def test_split_name_regime_and_dataset():
    assert results._split_name(NAME) == ("csp_lda", "within", "bnci2014_001")
    assert results._split_name("riemann_acm_cross_subject_bnci2014_001") == (
        "riemann_acm", "cross_subject", "bnci2014_001")
    assert results._split_name("fnirs_lda_cross_subject_shin2017_nback") == (
        "fnirs_lda", "cross_subject", "shin2017_nback")


def test_metrics_handles_both_schemas():
    assert results._metrics(_harness(ACC, KAPPA, ECE)) == {"acc": ACC, "kappa": KAPPA, "ece": ECE}
    assert results._metrics({"acc_mean": ACC, "kappa_mean": KAPPA}) == {"acc": ACC, "kappa": KAPPA, "ece": None}
    assert results._metrics({"nothing": 1}) is None


def test_collect_rounds_to_precision(tmp_path):
    row = results.collect(_write_run(tmp_path, NAME).parent)[NAME]
    assert row["acc"] == round(ACC, results._PRECISION)          # snapshot keeps _PRECISION decimals
    assert row["dataset"] == "bnci2014_001"


def test_record_upserts_and_preserves_others(tmp_path):
    out = tmp_path / "results.json"
    other = "riemann_within_bnci2014_001"
    a = _write_run(tmp_path, NAME, _harness(acc=0.5))
    _write_run(tmp_path, other, _harness(acc=0.7))
    assert results.record(a, out) == NAME
    assert results.record(tmp_path / other, out) == other
    assert set(json.loads(out.read_text())["runs"]) == {NAME, other}    # both kept

    (a / "aggregate.json").write_text(json.dumps(_harness(acc=0.55)))   # re-record a changed number
    results.record(a, out)
    runs = json.loads(out.read_text())["runs"]
    assert runs[NAME]["acc"] == 0.55 and runs[other]["acc"] == 0.7      # upsert in place, sibling untouched


def test_record_nonfatal_on_missing(tmp_path):
    assert results.record(tmp_path / "nope", tmp_path / "results.json") is None   # no aggregate -> None, no raise


def test_render_single_and_gap():
    runs = {"within": {"acc": ACC}, "cross": {"acc": CROSS_ACC}}
    assert sync_numbers._render(runs, "within.acc") == "0.598"                    # 3dp display of ACC
    assert sync_numbers._render(runs, "cross.acc-within.acc") == "−0.215"         # signed, unicode minus


def test_render_rejects_bad_term():
    with pytest.raises(KeyError):
        sync_numbers._render({"a": {"acc": ACC}}, "a.f1score")     # unknown field
    with pytest.raises(KeyError):
        sync_numbers._render({}, "missing.acc")                   # unknown run
