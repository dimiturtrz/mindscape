"""Equivalence-class tests for the README numbers-sync (results.py + sync_numbers.py).

Covers the pure logic — schema normalization, run-name parsing, marker rendering — without touching
the gitignored runs/ tree or the real README.
"""
import json

import pytest

from neuroscan.evaluation import results, sync_numbers


def test_split_name_regime_and_dataset():
    assert results._split_name("csp_lda_within_bnci2014_001") == ("csp_lda", "within", "bnci2014_001")
    assert results._split_name("riemann_acm_cross_subject_bnci2014_001") == (
        "riemann_acm", "cross_subject", "bnci2014_001")
    assert results._split_name("fnirs_lda_cross_subject_shin2017_nback") == (
        "fnirs_lda", "cross_subject", "shin2017_nback")


def test_metrics_handles_both_schemas():
    harness = {"fold_mean": {"acc": 0.6, "kappa": 0.4, "ece": 0.1}}
    align = {"acc_mean": 0.5, "kappa_mean": 0.3}
    assert results._metrics(harness) == {"acc": 0.6, "kappa": 0.4, "ece": 0.1}
    assert results._metrics(align) == {"acc": 0.5, "kappa": 0.3, "ece": None}
    assert results._metrics({"nothing": 1}) is None


def test_collect_reads_aggregates(tmp_path):
    d = tmp_path / "csp_lda_within_bnci2014_001"
    d.mkdir()
    (d / "aggregate.json").write_text(json.dumps(
        {"method": "csp_lda", "regime": "within", "n_classes": 4,
         "fold_mean": {"acc": 0.59761, "kappa": 0.4635, "ece": 0.1395}}))
    out = results.collect(tmp_path)
    row = out["csp_lda_within_bnci2014_001"]
    assert row["acc"] == 0.5976            # rounded to 4dp
    assert row["dataset"] == "bnci2014_001"


def _make_run(tmp_path, name, acc):
    d = tmp_path / name
    d.mkdir()
    (d / "aggregate.json").write_text(json.dumps(
        {"method": name.split("_")[0], "regime": "within", "n_classes": 4,
         "fold_mean": {"acc": acc, "kappa": 0.4, "ece": 0.1}}))
    return d


def test_record_upserts_and_preserves_others(tmp_path):
    out = tmp_path / "results.json"
    a = _make_run(tmp_path, "csp_lda_within_bnci2014_001", 0.5)
    b = _make_run(tmp_path, "riemann_within_bnci2014_001", 0.7)
    assert results.record(a, out) == "csp_lda_within_bnci2014_001"
    assert results.record(b, out) == "riemann_within_bnci2014_001"
    runs = json.loads(out.read_text())["runs"]
    assert set(runs) == {"csp_lda_within_bnci2014_001", "riemann_within_bnci2014_001"}   # both kept
    # re-record a changed number -> upsert in place, sibling untouched
    (a / "aggregate.json").write_text(json.dumps({"fold_mean": {"acc": 0.55, "kappa": 0.4, "ece": 0.1}}))
    results.record(a, out)
    runs = json.loads(out.read_text())["runs"]
    assert runs["csp_lda_within_bnci2014_001"]["acc"] == 0.55
    assert runs["riemann_within_bnci2014_001"]["acc"] == 0.7


def test_record_nonfatal_on_missing(tmp_path):
    assert results.record(tmp_path / "nope", tmp_path / "results.json") is None   # no aggregate -> None, no raise


def test_render_single_and_gap():
    runs = {
        "a": {"acc": 0.5976, "kappa": 0.4635},
        "b": {"acc": 0.3823, "kappa": 0.1764},
    }
    assert sync_numbers._render(runs, "a.acc") == "0.598"
    # within->cross gap: signed, unicode minus
    assert sync_numbers._render(runs, "b.acc-a.acc") == "−0.215"


def test_render_rejects_bad_term():
    with pytest.raises(KeyError):
        sync_numbers._render({"a": {"acc": 0.5}}, "a.f1score")
    with pytest.raises(KeyError):
        sync_numbers._render({}, "missing.acc")
