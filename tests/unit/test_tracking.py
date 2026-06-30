"""Tracking must be a guarded no-op when disabled — the harness can never depend on mlflow."""
from neuroscan import tracking


def test_disabled_tracking_is_noop(monkeypatch):
    monkeypatch.setenv("MINDSCAPE_NO_MLFLOW", "1")
    with tracking.run("mindscape", "test_run", params={"a": 1}, tags={"t": "x"}):
        tracking.metrics({"acc": 0.5})
        tracking.per_group("acc_subject", {"1": 0.6})
        tracking.set_tags({"k": "v"})
        tracking.artifact_json("x.json", {"ok": True})
    # no exception = pass


def test_metrics_outside_run_is_safe():
    # calling loggers with no active run must not raise
    tracking.metrics({"acc": 0.5})
    tracking.per_group("g", {"1": 0.1})
    tracking.artifact("nonexistent.json")
