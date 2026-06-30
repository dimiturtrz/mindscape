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


def test_save_model_sklearn_writes_joblib(tmp_path):
    # a sklearn estimator (the baseline kind, no `.net`) -> <name>.joblib in run_dir/models/
    import joblib
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    clf = LogisticRegression().fit(np.eye(4), [0, 1, 0, 1])
    path = tracking.save_model(clf, "model_csp_1", run_dir=tmp_path)
    assert path is not None and path.exists() and path.suffix == ".joblib"
    assert path.parent == tmp_path / "models"
    reloaded = joblib.load(path)                     # must round-trip
    assert reloaded.predict(np.eye(4)).shape == (4,)


def test_save_model_torch_writes_pt(tmp_path):
    # an object exposing `.net` (the BraindecodeClf shape) -> <name>.pt (whole module)
    torch = __import__("torch")

    class FakeClf:
        net = torch.nn.Linear(3, 2)

    path = tracking.save_model(FakeClf(), "model_eegnet_1", run_dir=tmp_path)
    assert path is not None and path.exists() and path.suffix == ".pt"
    reloaded = torch.load(path, weights_only=False)
    assert isinstance(reloaded, torch.nn.Linear)


def test_save_model_is_guarded_on_failure(tmp_path):
    # an unpicklable object must return None, not raise
    unpicklable = lambda x: x        # noqa: E731 — lambdas don't pickle
    assert tracking.save_model(unpicklable, "bad", run_dir=tmp_path) is None
