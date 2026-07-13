"""Cross-subject Riemannian transfer — the RPA ladder, reported BY REGIME.

Plain tangent-space + LR transfers badly across subjects (~0.36 LOSO): each subject's covariance cloud sits
at a different LOCATION on the SPD manifold — a domain shift, not a difference in the shared ERD contrast.
Riemannian Procrustes Analysis (Rodrigues 2019) aligns the domains in three steps; we report where each sits
on the deployability axis:

  ZERO-SHOT (no target labels — deployment-real):
    recenter        re-center each subject to the identity by its own Riemannian mean (Zanini 2018). [step 1]
    recenter_scale  + normalize each subject's DISPERSION around the mean.                            [+ step 2]
  CALIBRATED (a FEW labeled target trials — a short calibration session):
    rpa             full RPA: re-center + re-scale + re-ROTATE the target to align class structure.  [+ step 3]
    mdwm            Minimum Distance to Weighted Mean — source↔target class-mean interpolation.

The rotation (step 3) and MDWM are SUPERVISED — they need target labels. Those labels come from a DISJOINT
calibration split of the held-out subject (the `calib_frac` param), fit there, evaluated on the REMAINING blocks
only. Test labels are never touched — the same rigor as the per-subject-calibration ablation.

    python -m neuroscan.tasks.motor_imagery.align --exp mi_align_recenter        # zero-shot (the baseline fix)
    python -m neuroscan.tasks.motor_imagery.align --exp mi_align_rpa             # calibrated full RPA
    python -m neuroscan.tasks.motor_imagery.align --exp mi_align_recenter_acm    # zero-shot on ACM covariances

The method + calibration/ACM knobs live in experiments.yaml (`params:`); argv keeps only --exp (+ --set for
ad-hoc tweaks like `--set params.mdwm_lambda=1.0`) and the resource knob --jobs.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from pydantic import BaseModel
from pyriemann.estimation import Covariances
from sklearn.model_selection import StratifiedShuffleSplit

from baselines.eeg import transfer
from core import config
from core.reference import Reference
from core.data import splits, store
from core.data.eeg.base import EpochCfg
from core.features import Covariance
from neuroscan import tracking
from neuroscan.evaluation import metrics, results

logger = logging.getLogger(__name__)

_ZERO_SHOT = {"recenter", "recenter_scale"}
_CALIBRATED = {"rpa", "mdwm"}


class AlignConfig(BaseModel):
    """The transfer experiment's method + knobs. `method` = recenter|recenter_scale|rpa|mdwm; `calib_frac`/
    `seed` = the calibrated split; `mdwm_lambda` = MDWM source↔target tradeoff; `augment`/`order`/`lag` = the
    ACM time-delay-embedding of the covariances."""
    method: str
    calib_frac: float = 0.5
    seed: int = 0
    mdwm_lambda: float = 0.5
    augment: bool = False
    order: int = 4
    lag: int = 8


def _covariances(X, augment, order, lag, estimator="oas"):
    if augment:
        X = Covariance.time_delay_embed(X.astype(np.float64), order, lag)
    return Covariances(estimator=estimator).transform(X.astype(np.float64))


def _zero_shot_fold(s, train: transfer.Domain, test: transfer.Domain, cfg: AlignConfig):
    """Zero-shot: delegate the alignment + classifier to the transfer method, score ALL target."""
    probs = transfer.zero_shot_predict(train, transfer.Domain(test.cov),
                                       scale=(cfg.method == "recenter_scale"))
    return _row(s, test.labels, probs)


def _calibrated_fold(s, train: transfer.Domain, test: transfer.Domain, cfg: AlignConfig):
    """Calibrated: carve a stratified `calib_frac` of the held-out subject as the *only* labelled target data
    (the rest is the disjoint test set), hand it to the transfer method, score the disjoint remainder. Test
    labels never enter the fit — the split is the runner's leakage-free guarantee, the method just consumes it."""
    cal, ev = next(StratifiedShuffleSplit(1, train_size=cfg.calib_frac,
                                          random_state=cfg.seed).split(test.cov, test.labels))
    pred = transfer.calibrated_predict(cfg.method, train,
                                       transfer.Domain(test.cov[cal], test.labels[cal]),
                                       transfer.Domain(test.cov[ev]), cfg.mdwm_lambda)
    yev = test.labels[ev]
    row = {"fold": str(s), "n": int(len(ev)), "n_calib": int(len(cal)),
           "acc": metrics.accuracy(yev, pred), "kappa": metrics.kappa(yev, pred), "ece": 0.0}
    return row, None, yev


def _row(s, yte, probs):
    pred = probs.argmax(1)
    row = {"fold": str(s), "n": int(len(yte)), "acc": metrics.accuracy(yte, pred),
           "kappa": metrics.kappa(yte, pred), "ece": metrics.ece_from_probs(probs, yte)}
    return row, probs, yte


def _run_fold(s, tr, te, cfg: AlignConfig):
    """One LOSO fold — module-level so joblib ships it to a worker (folds are independent)."""
    Xtr, ytr = store.Store.gather(tr)
    Xte, yte = store.Store.gather(te)
    train = transfer.Domain(_covariances(Xtr, cfg.augment, cfg.order, cfg.lag), ytr, tr["subject"].to_numpy())
    test = transfer.Domain(_covariances(Xte, cfg.augment, cfg.order, cfg.lag), yte)
    if cfg.method in _ZERO_SHOT:
        return _zero_shot_fold(s, train, test, cfg)
    return _calibrated_fold(s, train, test, cfg)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="mi_align_recenter",
                    help="named transfer experiment in experiments.yaml (task: align)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=val",
                    help="ad-hoc override, e.g. --set method=mdwm --set params.mdwm_lambda=1.0")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel LOSO folds (joblib; -1 = all cores)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true", help="skip updating the committed results.json snapshot")
    args = ap.parse_args()

    exp = config.load_experiment(args.exp, args.overrides)
    dataset, method = exp.dataset, exp.method
    p = exp.params
    calib_frac = p.get("calib_frac", 0.5)
    augment = p.get("augment", False)
    fold_cfg = AlignConfig(method=method, calib_frac=calib_frac, seed=p.get("seed", 0),
                           mdwm_lambda=p.get("mdwm_lambda", 0.5), augment=augment,
                           order=p.get("order", 4), lag=p.get("lag", 8))

    cfg = EpochCfg(**exp.recipe)
    meta = store.Store.load(dataset, cfg)
    cov = "acm" if augment else "ts"
    regime = "calibrated" if method in _CALIBRATED else "zero_shot"
    name = f"riemann_{method}_{cov}"                # …_ts / …_acm always (keeps riemann_recenter_ts markers)
    logger.info(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · recipe {cfg.key()} · "
          f"{method} ({regime}, cov={cov})" + (f" · calib {calib_frac:.0%}" if regime == "calibrated" else ""))

    folds = list(splits.Splits.leave_one_subject_out(meta))
    logger.info(f"\n=== {name} · cross_subject · {dataset} ({len(folds)} folds, jobs={args.jobs}) ===")
    out_folds = Parallel(n_jobs=args.jobs)(
        delayed(_run_fold)(s, tr, te, fold_cfg) for s, tr, te in folds)

    rows, P, Y = [], [], []
    for row, probs, yte in sorted(out_folds, key=lambda r: r[0]["fold"]):
        rows.append(row)
        Y.append(yte)
        if probs is not None:
            P.append(probs)
        cal = f" calib={row['n_calib']}" if "n_calib" in row else ""
        logger.info(f"  {row['fold']:>6}  acc {row['acc']:.3f}  kappa {row['kappa']:.3f}  (n={row['n']}{cal})")

    acc = float(np.mean([r["acc"] for r in rows]))
    kap = float(np.mean([r["kappa"] for r in rows]))
    logger.info(f"  {'MEAN':>6}  acc {acc:.3f}  kappa {kap:.3f}   [{regime}]")
    logger.info("  vs reference: " + Reference.compare(acc, dataset, "cross_subject", "riemann"))
    logger.info("  (un-recentered riemann LOSO ~0.36; recenter ~0.50 — read the ladder against those)")

    out = Path(args.out) if args.out else Path("runs") / f"{name}_{dataset}"
    out.mkdir(parents=True, exist_ok=True)
    res = {"method": name, "regime": "cross_subject", "transfer_regime": regime,
           "calib_frac": calib_frac if regime == "calibrated" else None,
           "acc_mean": acc, "kappa_mean": kap, "per_fold": rows}
    (out / "aggregate.json").write_text(json.dumps(res, indent=2))
    with tracking.run("mindscape", f"{name}_cross",
                      params={"exp": args.exp, "method": name, "transfer_regime": regime,
                              "augment": augment, "calib_frac": calib_frac},
                      tags={"kind": "transfer", "regime": "cross_subject"}, run_dir=out):
        tracking.metrics({"acc_mean": acc, "kappa_mean": kap})
        tracking.per_group("acc_subject", {r["fold"]: r["acc"] for r in rows})
        tracking.artifact(out / "aggregate.json")
    if not args.no_record and results.record(out):
        logger.info(f"   recorded -> results.json ({out.name})")
    logger.info(f"-> {out}/aggregate.json")


if __name__ == "__main__":
    main()
