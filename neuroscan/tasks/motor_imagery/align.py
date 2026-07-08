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
only. Test labels are never touched — the same honesty as the per-subject-calibration ablation.

    python -m neuroscan.tasks.motor_imagery.align --exp mi_align_recenter        # zero-shot (the baseline fix)
    python -m neuroscan.tasks.motor_imagery.align --exp mi_align_rpa             # calibrated full RPA
    python -m neuroscan.tasks.motor_imagery.align --exp mi_align_recenter_acm    # zero-shot on ACM covariances

The method + calibration/ACM knobs live in experiments.yaml (`params:`); argv keeps only --exp (+ --set for
ad-hoc tweaks like `--set params.mdwm_lambda=1.0`) and the resource knob --jobs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.eeg import transfer
from core import config, reference
from core.data import splits, store
from core.data.eeg.base import EpochCfg
from core.features import time_delay_embed
from neuroscan import tracking
from neuroscan.evaluation import metrics

_ZERO_SHOT = {"recenter", "recenter_scale"}
_CALIBRATED = {"rpa", "mdwm"}


def _covariances(X, augment, order, lag, estimator="oas"):
    from pyriemann.estimation import Covariances
    if augment:
        X = time_delay_embed(X.astype(np.float64), order, lag)
    return Covariances(estimator=estimator).transform(X.astype(np.float64))


def _zero_shot_fold(s, Ctr, ytr, Cte, yte, groups, scale):
    """Zero-shot: delegate the alignment + classifier to the transfer method, score ALL target."""
    probs = transfer.zero_shot_predict(Ctr, ytr, groups, Cte, scale)
    return _row(s, yte, probs)


def _calibrated_fold(s, method, Ctr, ytr, Cte, yte, calib_frac, seed, mdwm_lambda=0.5):
    """Calibrated: carve a stratified `calib_frac` of the held-out subject as the *only* labelled target data
    (the rest is the disjoint test set), hand it to the transfer method, score the disjoint remainder. Test
    labels never enter the fit — the split is the runner's honesty guarantee, the method just consumes it."""
    from sklearn.model_selection import StratifiedShuffleSplit

    cal, ev = next(StratifiedShuffleSplit(1, train_size=calib_frac, random_state=seed).split(Cte, yte))
    pred = transfer.calibrated_predict(method, Ctr, ytr, Cte[cal], yte[cal], Cte[ev], mdwm_lambda)
    yev = yte[ev]
    row = {"fold": str(s), "n": int(len(ev)), "n_calib": int(len(cal)),
           "acc": metrics.accuracy(yev, pred), "kappa": metrics.kappa(yev, pred), "ece": 0.0}
    return row, None, yev


def _row(s, yte, probs):
    pred = probs.argmax(1)
    row = {"fold": str(s), "n": int(len(yte)), "acc": metrics.accuracy(yte, pred),
           "kappa": metrics.kappa(yte, pred), "ece": metrics.ece_from_probs(probs, yte)}
    return row, probs, yte


def _run_fold(s, tr, te, method, calib_frac, seed, augment, order, lag, mdwm_lambda=0.5):
    """One LOSO fold — module-level so joblib ships it to a worker (folds are independent)."""
    Xtr, ytr = store.gather(tr)
    Xte, yte = store.gather(te)
    Ctr = _covariances(Xtr, augment, order, lag)
    Cte = _covariances(Xte, augment, order, lag)
    if method in _ZERO_SHOT:
        return _zero_shot_fold(s, Ctr, ytr, Cte, yte, tr["subject"].to_numpy(),
                               scale=(method == "recenter_scale"))
    return _calibrated_fold(s, method, Ctr, ytr, Cte, yte, calib_frac, seed, mdwm_lambda)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="mi_align_recenter",
                    help="named transfer experiment in experiments.yaml (task: align)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="key=val",
                    help="ad-hoc override, e.g. --set method=mdwm --set params.mdwm_lambda=1.0")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel LOSO folds (joblib; -1 = all cores)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true", help="skip updating the committed results.json snapshot")
    args = ap.parse_args()

    from joblib import Parallel, delayed

    exp = config.load_experiment(args.exp, args.overrides)
    dataset, method = exp.dataset, exp.method
    p = exp.params
    calib_frac = p.get("calib_frac", 0.5)
    seed = p.get("seed", 0)
    mdwm_lambda = p.get("mdwm_lambda", 0.5)
    augment = p.get("augment", False)
    order, lag = p.get("order", 4), p.get("lag", 8)

    cfg = EpochCfg(**exp.recipe)
    meta = store.load(dataset, cfg)
    cov = "acm" if augment else "ts"
    regime = "calibrated" if method in _CALIBRATED else "zero_shot"
    name = f"riemann_{method}_{cov}"                # …_ts / …_acm always (keeps riemann_recenter_ts markers)
    print(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · recipe {cfg.key()} · "
          f"{method} ({regime}, cov={cov})" + (f" · calib {calib_frac:.0%}" if regime == "calibrated" else ""))

    folds = list(splits.leave_one_subject_out(meta))
    print(f"\n=== {name} · cross_subject · {dataset} ({len(folds)} folds, jobs={args.jobs}) ===")
    out_folds = Parallel(n_jobs=args.jobs)(
        delayed(_run_fold)(s, tr, te, method, calib_frac, seed, augment, order, lag, mdwm_lambda)
        for s, tr, te in folds)

    rows, P, Y = [], [], []
    for row, probs, yte in sorted(out_folds, key=lambda r: r[0]["fold"]):
        rows.append(row); Y.append(yte)
        if probs is not None:
            P.append(probs)
        cal = f" calib={row['n_calib']}" if "n_calib" in row else ""
        print(f"  {row['fold']:>6}  acc {row['acc']:.3f}  kappa {row['kappa']:.3f}  (n={row['n']}{cal})")

    acc = float(np.mean([r["acc"] for r in rows]))
    kap = float(np.mean([r["kappa"] for r in rows]))
    print(f"  {'MEAN':>6}  acc {acc:.3f}  kappa {kap:.3f}   [{regime}]")
    print("  vs reference: " + reference.compare(acc, dataset, "cross_subject", "riemann"))
    print("  (un-recentered riemann LOSO ~0.36; recenter ~0.50 — read the ladder against those)")

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
    from neuroscan.evaluation import results
    if not args.no_record and results.record(out):
        print(f"   recorded -> results.json ({out.name})")
    print(f"-> {out}/aggregate.json")


if __name__ == "__main__":
    main()
