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
calibration split of the held-out subject (`--calib-frac`), fit there, evaluated on the REMAINING blocks
only. Test labels are never touched — the same honesty as the per-subject-calibration ablation.

    python -m neuroscan.tasks.motor_imagery.align --method recenter          # zero-shot (the baseline fix)
    python -m neuroscan.tasks.motor_imagery.align --method rpa --calib-frac 0.5   # calibrated full RPA
    python -m neuroscan.tasks.motor_imagery.align --method mdwm --augment    # calibrated, on ACM covariances
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core import reference
from core.data import splits, store
from core.data.eeg.base import EpochCfg
from baselines import riemann
from neuroscan import tracking
from neuroscan.evaluation import metrics

_ZERO_SHOT = {"recenter", "recenter_scale"}
_CALIBRATED = {"rpa", "mdwm"}


def _covariances(X, augment, order, lag, estimator="oas"):
    from pyriemann.estimation import Covariances
    if augment:
        X = riemann._augment(X.astype(np.float64), order, lag)
    return Covariances(estimator=estimator).transform(X.astype(np.float64))


def _scale_to_identity(C, target_disp=1.0):
    """Normalize dispersion: after re-centering to the identity, stretch each covariance so the mean squared
    Riemannian distance to I equals `target_disp` (RPA step 2 — matches the domains' spread, not just their
    location). `C -> C**p` with p = sqrt(target_disp / current_dispersion)."""
    from pyriemann.utils.base import powm
    from pyriemann.utils.distance import distance_riemann
    eye = np.eye(C.shape[-1])
    disp = float(np.mean([distance_riemann(c, eye) ** 2 for c in C])) + 1e-12
    p = np.sqrt(target_disp / disp)
    return np.stack([powm(c, p) for c in C])


def _align_by_group(C, groups, scale):
    """Re-center (and optionally re-scale) each domain (subject) independently — the unsupervised, per-domain
    transforms, safe to apply to a labelled or unlabelled domain alike."""
    out = np.empty_like(C)
    for g in np.unique(groups):
        m = groups == g
        rc = riemann.recenter_covariances(C[m])
        out[m] = _scale_to_identity(rc) if scale else rc
    return out


def _zero_shot_fold(s, Ctr, ytr, Cte, yte, groups, scale):
    """Re-center (± scale) train per-subject + target unsupervised, tangent-space + LR, score ALL target."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    Ctr = _align_by_group(Ctr, groups, scale)
    Cte = _scale_to_identity(riemann.recenter_covariances(Cte)) if scale else riemann.recenter_covariances(Cte)
    clf = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(max_iter=500, C=1.0))
    clf.fit(Ctr, ytr)
    probs = np.asarray(clf.predict_proba(Cte), dtype=float)
    return _row(s, yte, probs)


def _calibrated_fold(s, method, Ctr, ytr, Cte, yte, calib_frac, seed, mdwm_lambda=0.5):
    """SUPERVISED transfer with a DISJOINT target calibration split. Carve a stratified `calib_frac` of the
    held-out subject as the *only* labelled target data; fit RPA-rotation / MDWM on source + that calib slice;
    predict — and score — the REMAINING (disjoint) target blocks. Test labels never enter the fit."""
    from sklearn.model_selection import StratifiedShuffleSplit

    from pyriemann.tangentspace import TangentSpace
    from pyriemann.transfer import MDWM, TLCenter, TLClassifier, TLRotate, TLScale, encode_domains
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    cal, ev = next(StratifiedShuffleSplit(1, train_size=calib_frac, random_state=seed).split(Cte, yte))
    Xf = np.concatenate([Ctr, Cte[cal]])
    yf = np.concatenate([ytr, yte[cal]]).astype(str)
    dom = np.array(["source"] * len(ytr) + ["target"] * len(cal))
    Xenc, yenc = encode_domains(Xf, yf, dom)
    Xev, _ = encode_domains(Cte[ev], yte[ev].astype(str), np.array(["target"] * len(ev)))

    if method == "mdwm":
        model = MDWM(domain_tradeoff=mdwm_lambda, target_domain="target")
    else:                                                            # full RPA + tangent-space LR
        base = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(max_iter=500, C=1.0))
        model = make_pipeline(TLCenter("target"), TLScale("target", centered_data=True),
                              TLRotate("target"), TLClassifier("target", base))
    model.fit(Xenc, yenc)
    pred = model.predict(Xev).astype(int)
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
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--method", default="recenter", choices=sorted(_ZERO_SHOT | _CALIBRATED),
                    help="recenter / recenter_scale (zero-shot) · rpa / mdwm (calibrated)")
    ap.add_argument("--calib-frac", type=float, default=0.5,
                    help="calibrated methods: fraction of the target subject used as the labelled calibration "
                         "split (the rest is the disjoint test set)")
    ap.add_argument("--seed", type=int, default=0, help="calibration-split seed")
    ap.add_argument("--mdwm-lambda", type=float, default=0.5,
                    help="MDWM domain tradeoff (0=source-only, 1=target-only). Sweep it to see the fragility.")
    ap.add_argument("--augment", action="store_true", help="time-delay-embed covariances (ACM) first")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--lag", type=int, default=8)
    ap.add_argument("--resample", type=float, default=128.0)
    ap.add_argument("--fmin", type=float, default=8.0)
    ap.add_argument("--fmax", type=float, default=32.0)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel LOSO folds (joblib; -1 = all cores)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true", help="skip updating the committed results.json snapshot")
    args = ap.parse_args()

    from joblib import Parallel, delayed

    cfg = EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax)
    meta = store.load(args.dataset, cfg)
    cov = "acm" if args.augment else "ts"
    regime = "calibrated" if args.method in _CALIBRATED else "zero_shot"
    name = f"riemann_{args.method}_{cov}"           # …_ts / …_acm always (keeps riemann_recenter_ts markers)
    print(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · recipe {cfg.key()} · "
          f"{args.method} ({regime}, cov={cov})" + (f" · calib {args.calib_frac:.0%}" if regime == "calibrated" else ""))

    folds = list(splits.leave_one_subject_out(meta))
    print(f"\n=== {name} · cross_subject · {args.dataset} ({len(folds)} folds, jobs={args.jobs}) ===")
    out_folds = Parallel(n_jobs=args.jobs)(
        delayed(_run_fold)(s, tr, te, args.method, args.calib_frac, args.seed, args.augment, args.order,
                            args.lag, args.mdwm_lambda)
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
    print("  vs reference: " + reference.compare(acc, args.dataset, "cross_subject", "riemann"))
    print("  (un-recentered riemann LOSO ~0.36; recenter ~0.50 — read the ladder against those)")

    out = Path(args.out) if args.out else Path("runs") / f"{name}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    res = {"method": name, "regime": "cross_subject", "transfer_regime": regime,
           "calib_frac": args.calib_frac if regime == "calibrated" else None,
           "acc_mean": acc, "kappa_mean": kap, "per_fold": rows}
    (out / "aggregate.json").write_text(json.dumps(res, indent=2))
    with tracking.run("mindscape", f"{name}_cross",
                      params={"method": name, "transfer_regime": regime, "augment": args.augment,
                              "calib_frac": args.calib_frac},
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
