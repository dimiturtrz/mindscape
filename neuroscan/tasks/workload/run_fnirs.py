"""Stage-2 entrypoint: decode fNIRS n-back workload through the SAME eval harness as EEG.

Different task/modality (Shin n-back, 3-class, hemodynamic), same spine — proving the harness is
modality-agnostic. The right decoder differs though: covariance methods (csp_lda, riemann) sit at chance
because fNIRS class info is in the HbO amplitude the covariance discards; `fnirs_lda` (mean+slope+peak ->
LDA) is the field-standard that actually reads it.

    python -m neuroscan.tasks.workload.run_fnirs --exp nback_fnirs_cross
    python -m neuroscan.tasks.workload.run_fnirs --exp nback_fnirs_within
    python -m neuroscan.tasks.workload.run_fnirs --exp nback_fnirs_cross --set recipe.h_freq=0.1

Config coordinates live in experiments.yaml (see `--exp`); argv stays sparse (`--set` for ad-hoc tweaks).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core import config
from core.data import store
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import harness, results

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="nback_fnirs_cross",
                    help="named experiment in experiments.yaml")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=val",
                    help="ad-hoc override, e.g. --set recipe.h_freq=0.1 --set regime=within")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true",
                    help="skip updating the committed results.json snapshot (scratch/experimental runs)")
    args = ap.parse_args()

    exp = config.load_experiment(args.exp, args.overrides)
    dataset, method, regime = exp.dataset, exp.method, exp.regime
    cfg = FnirsCfg(**exp.recipe)
    meta = store.load(dataset, cfg)
    n_classes = int(meta["label_id"].max()) + 1
    chance = 1.0 / n_classes
    logger.info(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · "
          f"{n_classes} classes {sorted(meta['label'].unique().to_list())} · recipe {cfg.key()}")

    test_sessions = [exp.test_session] if (regime == "within" and exp.test_session) else ()
    folds = harness.folds_for(meta, regime, test_sessions=test_sessions)
    fit_fn, score_fn = models.get_method(method)

    run_dir = Path(args.out) if args.out else Path("runs") / f"{method}_{regime}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\n=== {method} · {regime} · {dataset} ({len(folds)} folds, chance {chance:.3f}) ===")
    n_jobs = -1 if method in {"csp_lda", "riemann", "riemann_acm", "fnirs_lda"} else 1
    method_obj = harness.Method(method, fit_fn, score_fn, n_classes, regime)
    res = harness.run(method_obj, folds, n_jobs=n_jobs,
                      tracking_cfg=harness.TrackConfig(
                          params={"exp": args.exp, "method": method, "regime": regime, "dataset": dataset,
                                  "modality": "fnirs"}, run_dir=run_dir))
    (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
    if not args.no_record and results.record(run_dir):
        logger.info(f"   recorded -> results.json ({run_dir.name})")
    fm = res["fold_mean"]
    logger.info(f"\nfold-mean acc {fm['acc']:.3f} | kappa {fm['kappa']:.3f} | ece {fm['ece']:.3f}  "
          f"(chance {chance:.3f})")
    logger.info(f"-> {run_dir}/aggregate.json")


if __name__ == "__main__":
    main()
