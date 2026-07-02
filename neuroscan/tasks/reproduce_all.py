"""Regenerate the canonical decode results that back the README (results.json) — in ONE process (imports
the heavy stack once) with parallel folds. Use after a protocol/harness change so every committed number
is refreshed consistently, instead of firing off N cold `uv run` processes.

    python -m neuroscan.tasks.reproduce_all              # all canonical runs
    python -m neuroscan.tasks.reproduce_all --cross-only # just the cross-subject set

Recentering (the transfer fix) has its own parallel entrypoint — run `align --exp mi_align_recenter`
(and `--exp mi_align_recenter_acm` for the ACM variant) after.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import config
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import harness, results

_BASELINES = {"csp_lda", "riemann", "riemann_acm", "fnirs_lda"}


def _canonical_runs():
    """The harness runs the README cites, straight from experiments.yaml (task: decode | fnirs) — one source
    of truth. align/fusion carry their own aggregation + entrypoints, so `align` / `run_fusion` regenerate
    those, not this. Returns (dataset, method, regime, cfg, test_session) tuples."""
    runs = []
    for name in config.experiment_names():
        exp = config.load_experiment(name)
        if exp.task not in ("decode", "fnirs"):
            continue
        cfg = FnirsCfg(**exp.recipe) if exp.task == "fnirs" else EpochCfg(**exp.recipe)
        runs.append((exp.dataset, exp.method, exp.regime, cfg, exp.test_session))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-only", action="store_true", help="skip within-subject runs")
    args = ap.parse_args()

    runs = [r for r in _canonical_runs() if not (args.cross_only and r[2] == "within")]
    for dataset, method, regime, cfg, test_session in runs:
        meta = store.load(dataset, cfg)
        n_classes = int(meta["label_id"].max()) + 1
        tsess = [test_session] if (regime == "within" and test_session) else ()
        folds = harness.folds_for(meta, regime, test_sessions=tsess)
        fit_fn, score_fn = models.get_method(method)
        n_jobs = -1 if method in _BASELINES else 1                # CPU baselines parallelize folds
        run_dir = Path("runs") / f"{method}_{regime}_{dataset}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {method} · {regime} · {dataset} ({len(folds)} folds, jobs {n_jobs}) ===")
        res = harness.run(method, fit_fn, score_fn, folds, n_classes, regime=regime,
                          params={"method": method, "regime": regime, "dataset": dataset},
                          run_dir=run_dir, n_jobs=n_jobs)
        (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
        results.record(run_dir)
    print("\nreproduce_all done — regenerated results.json. Run `align --exp mi_align_recenter` (+ "
          "`--exp mi_align_recenter_acm`) for recentering, then `sync_numbers` to update the README.")


if __name__ == "__main__":
    main()
