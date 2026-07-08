"""Canonical results snapshot — the committed bridge between local runs and the README.

The authoritative numbers live in `runs/<name>/aggregate.json` (gitignored, one per run) and, for the
cross-run UI, in mlflow. Neither is in the repo, so a fresh clone / CI can't read them — and the README
tables were hand-typed, which is how they drift (a claim said "not implemented" while the table showed the
number). This script closes that loop: it scans the local run aggregates and writes ONE committed
`results.json` snapshot. `sync_numbers.py` then injects those numbers into the README between markers, so
the prose can't drift from the measured result.

    uv run python -m neuroscan.evaluation.results          # rebuild-all: scan runs/ -> results.json

Three separated layers, each one responsibility:
  1. train/eval  -> writes runs/<name>/aggregate.json          (per-run, authoritative; gitignored)
  2. record      -> `record(run_dir)` merges ONE finished run into the committed results.json snapshot
                    (called automatically at the end of each experiment entrypoint; --no-record opts out)
  3. sync        -> sync_numbers.py injects results.json into the README (separate, periodic/deliberate)

So numbers update on training (layer 2), but the README only changes when you choose to sync (layer 3).
`write()` here is the layer-1.5 rebuild-all fallback (rescans every run). `results.json` is a *snapshot*,
not the source of truth. Two aggregate schemas exist and both are normalized:
  - harness runs:  {"method","regime","n_classes","fold_mean":{"acc","kappa","ece"}}
  - align.py runs: {"method","regime","acc_mean","kappa_mean","per_fold":[...]}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import REPO

logger = logging.getLogger(__name__)

_RUNS = REPO / "runs"
_OUT = REPO / "results.json"
_PRECISION = 4          # decimals kept in the snapshot (display rounding is sync_numbers' job)
_NOTE = ("Committed snapshot of local run aggregates. Do not hand-edit — it is updated automatically after "
         "training (record) or rebuilt with `python -m neuroscan.evaluation.results`; run `sync_numbers` to "
         "push these into the README.")

# datasets whose run-dir suffix we recognize (longest match first so shin2017_nback beats bnci2014_001)
_DATASETS = ("bnci2014_001", "shin2017_nback", "shin2017")


def _split_name(name: str) -> tuple[str, str, str]:
    """runs/<method>_<regime>_<dataset> -> (method, regime, dataset). Regime is one token here
    (`within` / `cross_subject` / `cross_session`); dataset is the recognized suffix."""
    dataset = next((d for d in _DATASETS if name.endswith(d)), "")
    stem = name[: -(len(dataset) + 1)] if dataset else name
    for regime in ("cross_subject", "cross_session", "within"):
        if stem.endswith(regime):
            return stem[: -(len(regime) + 1)], regime, dataset
    return stem, "", dataset


def read_metrics(agg: dict) -> dict | None:
    """Pull (acc, kappa, ece) from either aggregate schema. None if neither present.

    The one place that knows both run-aggregate shapes; reused by tracking.backfill so the schema
    knowledge isn't duplicated."""
    fm = agg.get("fold_mean")                                # harness schema: {"acc","kappa","ece"} or absent
    if fm and "acc" in fm:
        return {"acc": fm["acc"], "kappa": fm.get("kappa"), "ece": fm.get("ece")}
    if "acc_mean" in agg:                                    # align.py schema
        return {"acc": agg["acc_mean"], "kappa": agg.get("kappa_mean"), "ece": agg.get("ece_mean")}
    return None


def _row(name: str, agg: dict) -> dict | None:
    """Normalize one aggregate.json -> a snapshot row, or None if it has no usable metrics."""
    m = read_metrics(agg)
    if m is None:
        return None
    method, regime, dataset = _split_name(name)
    # fusion runs add dict[str, float] blocks — a per-role breakdown (eeg/fnirs/late/feature) and the
    # complementarity + aggregation-sweep scalars — pass them through as flat marker fields. Absent on
    # non-fusion runs (-> {}); comp wins on any key overlap with the sweep.
    extra = {**agg.get("per_role_mean", {}), **agg.get("aggregation", {}), **agg.get("complementarity", {})}
    return {
        "method": agg.get("method", method),
        "regime": agg.get("regime", regime),
        "dataset": dataset,
        "n_classes": agg.get("n_classes"),
        # metrics: kappa/ece are None on fusion runs (no per-fold kappa) — keep None, round the rest
        **{k: (round(v, _PRECISION) if v is not None else None) for k, v in m.items()},
        **{k: round(v, _PRECISION) for k, v in extra.items()},
    }


def collect(runs_dir: Path = _RUNS) -> dict:
    """Scan runs/*/aggregate.json -> {run_name: {method,regime,dataset,n_classes,acc,kappa,ece}}."""
    out: dict[str, dict] = {}
    for agg_path in sorted(runs_dir.glob("*/aggregate.json")):
        row = _row(agg_path.parent.name, json.loads(agg_path.read_text()))
        if row is not None:
            out[agg_path.parent.name] = row
    return out


def _dump(runs: dict, out_path: Path) -> Path:
    out_path.write_text(json.dumps({"_note": _NOTE, "runs": dict(sorted(runs.items()))}, indent=2) + "\n")
    return out_path


def write(out_path: Path = _OUT, runs_dir: Path = _RUNS) -> Path:
    """Rebuild-all: rescan every run and overwrite the snapshot."""
    return _dump(collect(runs_dir), out_path)


def record(run_dir: Path | str, out_path: Path = _OUT) -> str | None:
    """Layer 2 — merge ONE finished run's aggregate into the committed snapshot (upsert by run name).

    Called at the end of an experiment entrypoint, after runs/<name>/aggregate.json is written. Best-effort
    and non-fatal: a bad/absent aggregate just returns None rather than breaking the training run. Returns
    the run name on success. Only the named run is touched — other rows are preserved."""
    try:
        run_dir = Path(run_dir)
        agg_path = run_dir / "aggregate.json"
        if not agg_path.exists():
            return None
        row = _row(run_dir.name, json.loads(agg_path.read_text()))
        if row is None:
            return None
        payload = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}
        runs = payload.get("runs", {})
        runs[run_dir.name] = row
        _dump(runs, out_path)
        return run_dir.name
    except (OSError, ValueError):
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    p = write()
    n = len(json.loads(p.read_text())["runs"])
    logger.info(f"wrote {p.relative_to(REPO)} — {n} run(s)")
