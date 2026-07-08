"""Regenerate the canonical decode results that back the README (results.json) — in ONE process (imports
the heavy stack once) with parallel folds. Use after a protocol/harness change so every committed number
is refreshed consistently, instead of firing off N cold `uv run` processes.

    python -m neuroscan.tasks.reproduce_all                     # all canonical runs
    python -m neuroscan.tasks.reproduce_all --cross-only        # just the cross-subject set
    python -m neuroscan.tasks.reproduce_all --only a,b,c        # only these named experiments (one process)

Recentering (the transfer fix) has its own parallel entrypoint — run `align --exp mi_align_recenter`
(and `--exp mi_align_recenter_acm` for the ACM variant) after.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core import config
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import harness, results
from neuroscan.models.decoders import MODELS

logger = logging.getLogger(__name__)


def _canonical_runs():
    """The harness runs the README cites, straight from experiments.yaml (task: decode | fnirs) — one source
    of truth. align/fusion carry their own aggregation + entrypoints, so `align` / `run_fusion` regenerate
    those, not this. Returns (name, dataset, method, regime, cfg, test_session) tuples."""
    runs = []
    for name in config.experiment_names():
        exp = config.load_experiment(name)
        if exp.task not in ("decode", "fnirs"):
            continue
        cfg = FnirsCfg(**exp.recipe) if exp.task == "fnirs" else EpochCfg(**exp.recipe)
        runs.append((name, exp.dataset, exp.method, exp.regime, cfg, exp.test_session))
    return runs


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-only", action="store_true", help="skip within-subject runs")
    ap.add_argument("--only", default=None,
                    help="comma-separated experiment names — run only these (still one process, folds parallel)")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    runs = _canonical_runs()
    if only:
        runs = [r for r in runs if r[0] in only]
        missing = only - {r[0] for r in runs}
        if missing:
            raise SystemExit(f"--only names not decode/fnirs experiments: {sorted(missing)}")
    runs = [r for r in runs if not (args.cross_only and r[3] == "within")]
    for name, dataset, method, regime, cfg, test_session in runs:
        meta = store.load(dataset, cfg)
        n_classes = int(meta["label_id"].max()) + 1
        tsess = [test_session] if (regime == "within" and test_session) else ()
        folds = harness.folds_for(meta, regime, test_sessions=tsess)
        fit_fn, score_fn = models.get_method(method, fs=getattr(cfg, "resample", None))
        n_jobs = 1 if method in MODELS else -1                    # classical baselines parallelize folds; nets don't
        run_dir = Path("runs") / f"{method}_{regime}_{dataset}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n=== {name} · {method} · {regime} · {dataset} ({len(folds)} folds, jobs {n_jobs}) ===")
        method_obj = harness.Method(method, fit_fn, score_fn, n_classes, regime)
        res = harness.run(method_obj, folds, n_jobs=n_jobs,
                          tracking_cfg=harness.TrackConfig(
                              params={"exp": name, "method": method, "regime": regime, "dataset": dataset},
                              run_dir=run_dir))
        (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
        results.record(run_dir)
        logger.info(f"  -> acc {res['fold_mean']['acc']:.3f}  kappa {res['fold_mean']['kappa']:.3f}  "
              f"(chance {1/n_classes:.3f})")
    logger.info("\nreproduce_all done — regenerated results.json. Run `align --exp mi_align_recenter` (+ "
          "`--exp mi_align_recenter_acm`) for recentering, then `sync_numbers` to update the README.")


if __name__ == "__main__":
    main()
