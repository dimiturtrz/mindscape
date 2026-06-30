"""Optional MLflow tracking — a thin, GUARDED layer (mirrors the siblings').

If mlflow is absent or MINDSCAPE_NO_MLFLOW is set, every call is a no-op, so the harness never depends
on it. It's the cross-run comparison UI (`mlflow ui`), not the source of truth — the aggregate.json
artifact each run writes stays authoritative.

    with tracking.run("mindscape", method, params={...}):
        tracking.metrics({"acc_mean": 0.72})
        tracking.per_group("acc_subject", {"1": 0.81, ...})
        tracking.artifact_json("aggregate.json", res)
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MLRUNS = _ROOT / "mlruns"
_DB_URI = f"sqlite:///{(_ROOT / 'mlflow.db').as_posix()}"

_active = None   # the live mlflow module while a run is open, else None


def _mlflow():
    if os.environ.get("MINDSCAPE_NO_MLFLOW"):
        return None
    try:
        import mlflow
    except ImportError:
        return None
    return mlflow


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


@contextmanager
def run(experiment: str, run_name: str, params: dict | None = None):
    """Open a tracked run (local mlruns/). No-op context if tracking is off; never breaks the caller."""
    global _active
    mlflow = _mlflow()
    if mlflow is None:
        yield
        return
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_DB_URI)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        if params:
            mlflow.log_params(_flat(params))
        _active = mlflow
        yield
    except Exception:
        yield
    finally:
        try:
            if _active is not None:
                _active.end_run()
        except Exception:
            pass
        _active = None


def metrics(d: dict) -> None:
    if _active is None:
        return
    for k, v in d.items():
        try:
            _active.log_metric(k, float(v))
        except Exception:
            pass


def per_group(prefix: str, d: dict) -> None:
    """Log a per-group scalar dict as <prefix>_<group> (e.g. acc_subject_1)."""
    metrics({f"{prefix}_{g}": v for g, v in d.items()})


def artifact_json(name: str, obj) -> None:
    if _active is None:
        return
    try:
        p = Path(tempfile.gettempdir()) / name
        p.write_text(json.dumps(obj, indent=2))
        _active.log_artifact(str(p))
    except Exception:
        pass
