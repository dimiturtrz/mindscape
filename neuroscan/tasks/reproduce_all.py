"""Regenerate the canonical decode results that back the README (results.json) — in ONE process (imports
the heavy stack once) with parallel folds. Use after a protocol/harness change so every committed number
is refreshed consistently, instead of firing off N cold `uv run` processes.

    python -m neuroscan.tasks.reproduce_all              # all canonical runs
    python -m neuroscan.tasks.reproduce_all --cross-only # just the cross-subject set

Recentering (the transfer fix) has its own parallel entrypoint — run `align` / `align --augment` after.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import harness, results

_E = dict(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)   # EEG n-back recipe

# (dataset, method, regime, cfg, test_session) — the canonical set the README cites
_RUNS = [
    ("bnci2014_001",       "csp_lda",     "within",              EpochCfg(), "1test"),
    ("bnci2014_001",       "csp_lda",     "cross_subject",       EpochCfg(), None),
    ("bnci2014_001",       "riemann",     "within",              EpochCfg(), "1test"),
    ("bnci2014_001",       "riemann",     "cross_subject",       EpochCfg(), None),
    ("bnci2014_001",       "riemann_acm", "cross_subject",       EpochCfg(), None),
    ("shin2017_nback",     "fnirs_lda",   "cross_subject",       FnirsCfg(), None),
    ("shin2017_nback",     "fnirs_lda",   "cross_subject_kfold", FnirsCfg(), None),
    ("shin2017_nback",     "fnirs_lda",   "within",              FnirsCfg(), "2"),
    ("shin2017_nback_eeg", "csp_lda",     "cross_subject",       EpochCfg(**_E), None),
    ("shin2017_nback_eeg", "csp_lda",     "cross_subject_kfold", EpochCfg(**_E), None),
    ("shin2017_nback_eeg", "csp_lda",     "within",              EpochCfg(**_E), "2"),
    ("shin2017_nback_eeg", "riemann",     "cross_subject",       EpochCfg(**_E), None),
    ("shin2017_nback_eeg", "riemann",     "cross_subject_kfold", EpochCfg(**_E), None),
    ("shin2017_nback_eeg", "riemann",     "within",              EpochCfg(**_E), "2"),
]

_BASELINES = {"csp_lda", "riemann", "riemann_acm", "fnirs_lda"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-only", action="store_true", help="skip within-subject runs")
    args = ap.parse_args()

    runs = [r for r in _RUNS if not (args.cross_only and r[2] == "within")]
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
    print("\nreproduce_all done — regenerated results.json. Run `align` + `align --augment` for recentering,"
          " then `sync_numbers` to update the README.")


if __name__ == "__main__":
    main()
