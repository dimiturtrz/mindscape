"""MLflow experiment tracking — a thin, GUARDED layer (mirrors the siblings').

Local sqlite backend (mlflow.db) + mlruns/ artifact store. If MINDSCAPE_NO_MLFLOW is set, every call is a
no-op, so the harness never depends on it. It's the cross-run comparison UI
(`mlflow ui --backend-store-uri sqlite:///mlflow.db`), not the source of truth — the runs/<name>/
aggregate.json each run writes stays authoritative.

    with tracking.run("mindscape", name, params={...}, tags={...}, run_dir="runs/x"):
        tracking.metrics({"acc_mean": 0.58})
        tracking.per_group("acc_subject", {"1": 0.75, ...})
        tracking.artifact_json("aggregate.json", res)     # or tracking.artifact("runs/x/aggregate.json")

`run_dir` enables resume: a later step (quantize / calibrate) reopening the same run_dir logs INTO the
train run, so the UI shows the Stage-2 numbers next to the decode numbers — the sibling pattern.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import joblib
import mlflow
import torch

from core.config import REPO
from neuroscan.evaluation import results

logger = logging.getLogger(__name__)

_MLRUNS = REPO / "mlruns"
_DB_URI = f"sqlite:///{(REPO / 'mlflow.db').as_posix()}"

# Tracking is best-effort telemetry: it must never break a run, but it should never hide a real bug either.
# So catch the failures mlflow/IO/torch-save actually raise and log them; anything else propagates.
_TRACK_ERRORS = (mlflow.exceptions.MlflowException, OSError, RuntimeError)
_SAVE_ERRORS = (OSError, RuntimeError, TypeError, pickle.PicklingError)   # torch.save / joblib.dump failures

_active = None   # the live mlflow module while a run is open, else None


def _disabled() -> bool:
    """The one off-switch (MINDSCAPE_NO_MLFLOW): tests + fast local runs skip tracking. mlflow is a hard
    dep now, so there's no 'is it installed?' fallback — it either logs or is explicitly disabled."""
    return bool(os.environ.get("MINDSCAPE_NO_MLFLOW"))


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
def run(experiment: str, run_name: str, params: dict | None = None, tags: dict | None = None,
        run_dir: str | Path | None = None):
    """Open a tracked run (local mlruns/). No-op context if tracking is off; never breaks the caller.
    If `run_dir` holds a .mlflow_run_id, resume that run (log into it); else start fresh and persist the id."""
    global _active
    if _disabled():
        yield
        return
    try:
        _MLRUNS.mkdir(exist_ok=True)
        mlflow.set_tracking_uri(_DB_URI)
        mlflow.set_experiment(experiment)
        try:
            mlflow.enable_system_metrics_logging()      # GPU/CPU/mem (psutil + pynvml if present)
        except _TRACK_ERRORS as exc:
            logger.debug(f"tracking: {exc}")
        idf = Path(run_dir) / ".mlflow_run_id" if run_dir else None
        rid = idf.read_text().strip() if (idf and idf.exists()) else None
        if rid:
            mlflow.start_run(run_id=rid)                 # resume — don't re-log params
        else:
            mlflow.start_run(run_name=run_name)
            if params:
                mlflow.log_params(_flat(params))
            if idf:
                idf.parent.mkdir(parents=True, exist_ok=True)
                idf.write_text(mlflow.active_run().info.run_id)
        for k, v in (tags or {}).items():
            mlflow.set_tag(k, str(v))
        _active = mlflow
        yield
    except _TRACK_ERRORS as exc:
        logger.debug(f"tracking: {exc}")
        yield
    finally:
        try:
            if _active is not None:
                _active.end_run()
        except _TRACK_ERRORS as exc:
            logger.debug(f"tracking: {exc}")
        _active = None


def metrics(d: dict, step: int | None = None) -> None:
    if _active is None:
        return
    for k, v in d.items():
        try:
            _active.log_metric(k, float(v), step=step)
        except _TRACK_ERRORS as exc:
            logger.debug(f"tracking: {exc}")


def per_group(prefix: str, d: dict) -> None:
    """Log a per-group scalar dict as <prefix>_<group> (e.g. acc_subject_1)."""
    metrics({f"{prefix}_{g}": v for g, v in d.items()})



def artifact(path: str | Path) -> None:
    """Log an existing file as a run artifact."""
    if _active is None:
        return
    try:
        if Path(path).exists():
            _active.log_artifact(str(path))
    except _TRACK_ERRORS as exc:
        logger.debug(f"tracking: {exc}")


def artifact_json(name: str, obj) -> None:
    if _active is None:
        return
    try:
        p = Path(tempfile.gettempdir()) / name
        p.write_text(json.dumps(obj, indent=2))
        _active.log_artifact(str(p))
    except _TRACK_ERRORS as exc:
        logger.debug(f"tracking: {exc}")


def save_model(clf, name: str, run_dir: str | Path | None = None) -> Path | None:
    """Persist a trained model (best-effort, guarded) and log it as a run artifact.

    Handles both decoder kinds behind the harness contract:
      - braindecode nets (a `BraindecodeClf` with a `.net` torch module) -> `<name>.pt` (whole module,
        reloadable via torch.load)
      - sklearn baselines (CSP+LDA, Riemann pipelines) -> `<name>.joblib`

    Writes into `run_dir/models/` when given (so runs/<name>/ stays the authoritative store, mlflow or
    not), then logs the file to the active MLflow run if one is open. Returns the path (or None on failure)."""
    try:
        out_dir = (Path(run_dir) / "models") if run_dir else Path(tempfile.gettempdir())
        out_dir.mkdir(parents=True, exist_ok=True)
        net = getattr(clf, "net", None)
        if net is not None:                                  # torch decoder
            path = out_dir / f"{name}.pt"
            torch.save(net, path)
        else:                                                # sklearn pipeline (baseline)
            path = out_dir / f"{name}.joblib"
            joblib.dump(clf, path)
    except _SAVE_ERRORS as exc:
        logger.debug(f"save_model: {exc}")
        return None
    artifact(path)                                           # no-op if no active run
    return path


def backfill(experiment: str = "mindscape") -> None:
    """One-shot: log existing runs/<name>/aggregate.json as runs, so the UI has history.
    Skips runs already tracked (have .mlflow_run_id).  `python -m neuroscan.tracking`."""
    n = 0
    for aj in sorted((REPO / "runs").glob("**/aggregate.json")):
        rd = aj.parent
        if (rd / ".mlflow_run_id").exists():
            continue
        res = json.loads(aj.read_text())
        with run(experiment, rd.name, params={k: res.get(k) for k in ("method", "regime", "n_classes")},
                 run_dir=rd):
            metrics({f"{k}_mean": v for k, v in (results.read_metrics(res) or {}).items() if v is not None})
            artifact(aj)
        n += 1
        logger.info(f"backfilled {rd.name}")
    logger.info(f"backfilled {n} run(s)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    backfill()
