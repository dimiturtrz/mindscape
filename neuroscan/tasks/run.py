"""Entrypoint: decode a registered EEG dataset under a chosen regime, through the eval harness. Class count
is read from the data (not assumed 4), so it serves motor imagery AND the 3-class EEG n-back workload.

The run coordinates (dataset, method, regime, band/window recipe) live in experiments.yaml, addressed by
name — argv stays sparse. Pick one with `--exp`; tweak ad-hoc with `--set` (dotlist over the config):

    python -m neuroscan.tasks.run --exp mi_csp_within
    python -m neuroscan.tasks.run --exp mi_riemann_cross
    python -m neuroscan.tasks.run --exp nback_eeg_riemann_cross          # EEG n-back workload (Table B)
    python -m neuroscan.tasks.run --exp mi_csp_within --set method=atcnet --set recipe.resample=250

Methods: csp_lda + riemann + riemann_acm (the classical baselines) and the braindecode decoders (eegnet,
shallow_fbcsp, deep4, atcnet, eegconformer) — commodity → near-SOTA. The within/cross_subject contrast IS
the headline: the gap between them is the honest out-of-distribution number. For the cross-subject *fix*
(Riemannian re-centering), see `motor_imagery/align.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from core import config, reference
from core.data import store
from core.data.eeg.base import EpochCfg
from neuroscan import models
from neuroscan.evaluation import harness, modelcard, results

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _n in ("mne", "moabb", "braindecode"):
        logging.getLogger(_n).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="mi_csp_within",
                    help="named experiment in experiments.yaml (see that file for the list)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=val",
                    help="ad-hoc override of the config, e.g. --set method=riemann --set recipe.resample=250")
    ap.add_argument("--out", default=None, help="write aggregate.json here")
    ap.add_argument("--no-record", action="store_true",
                    help="skip updating the committed results.json snapshot (use for scratch/experimental runs)")
    args = ap.parse_args()

    exp = config.load_experiment(args.exp, args.overrides)
    dataset, method, regime = exp.dataset, exp.method, exp.regime
    cfg = EpochCfg(**exp.recipe)
    meta = store.load(dataset, cfg)
    n_classes = int(meta["label_id"].max()) + 1                  # derived from data, not assumed 4-class
    logger.info(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · {n_classes} classes · "
          f"sessions {sorted(meta['session'].unique().to_list())} · recipe {cfg.key()}")

    test_sessions = [exp.test_session] if (regime == "within" and exp.test_session) else ()
    folds = harness.folds_for(meta, regime, test_sessions=test_sessions)
    fit_fn, score_fn = models.get_method(method, fs=cfg.resample)

    run_dir = Path(args.out) if args.out else Path("runs") / f"{method}_{regime}_{dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # classical baselines are CPU + fold-independent -> parallelize folds; GPU nets stay on one device
    n_jobs = -1 if method in {"csp_lda", "riemann", "riemann_acm", "fnirs_lda"} else 1
    logger.info(f"\n=== {method} · {regime} · {dataset} ({len(folds)} folds, jobs {n_jobs}) ===")
    res = harness.run(method, fit_fn, score_fn, folds, n_classes, regime=regime,
                      params={"exp": args.exp, "method": method, "regime": regime,
                              "dataset": dataset, "resample": cfg.resample},
                      run_dir=run_dir, n_jobs=n_jobs)

    out = run_dir / "aggregate.json"
    out.write_text(json.dumps(res, indent=2))
    modelcard.write(res, dataset, regime, run_dir / "CARD.md")
    if not args.no_record and results.record(run_dir):
        logger.info(f"   recorded -> results.json ({run_dir.name})")
    ref_regime = "within_subject" if regime == "within" else "cross_subject"
    logger.info(f"\nfold-mean acc {res['fold_mean']['acc']:.3f} | pooled acc {res['pooled']['acc']:.3f} "
          f"| ece {res['fold_mean']['ece']:.3f}  (chance {1.0 / n_classes:.3f})")
    logger.info("  vs reference: " + reference.compare(res["fold_mean"]["acc"], dataset, ref_regime, method))
    logger.info(f"-> {out}")


if __name__ == "__main__":
    main()
