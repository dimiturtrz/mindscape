"""Cross-subject Riemannian transfer via manifold re-centering — the documented fix for the LOSO collapse.

Plain tangent-space + LR transfers badly across subjects (~0.357 LOSO) because each subject's covariance
cloud sits at a different LOCATION on the SPD manifold — a domain shift, not a difference in the shared
ERD contrast. This runner re-centers every subject (train AND the unlabeled target) to the identity by a
congruence transform with their own Riemannian mean (Zanini et al. 2018), then runs the same tangent-space
classifier. Target re-centering is unsupervised (uses only the target's trials, no labels) -> deployable.

    python -m neuroscan.experiments.align                 # recentered tangent space, LOSO
    python -m neuroscan.experiments.align --augment       # recentered ACM (time-delay covariances)

Compare the printed mean against the un-recentered LOSO baseline (riemann ~0.357) to read the transfer gain.
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


def _covariances(X, augment, order, lag, estimator="oas"):
    from pyriemann.estimation import Covariances
    if augment:
        X = riemann._augment(X.astype(np.float64), order, lag)
    return Covariances(estimator=estimator).transform(X.astype(np.float64))


def _recenter_by_group(C, groups):
    """Re-center each domain (subject) separately to the identity, in place over a copy."""
    out = np.empty_like(C)
    for g in np.unique(groups):
        m = groups == g
        out[m] = riemann.recenter_covariances(C[m])
    return out


def _run_fold(s, tr, te, augment, order, lag):
    """One LOSO fold: re-center train (per subject) + target (unsupervised), tangent-space + LR, score.
    Module-level so joblib can ship it to a worker process — folds are independent, so they parallelize."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    Xtr, ytr = store.gather(tr)
    Xte, yte = store.gather(te)
    groups = tr["subject"].to_numpy()
    Ctr = _recenter_by_group(_covariances(Xtr, augment, order, lag), groups)
    Cte = riemann.recenter_covariances(_covariances(Xte, augment, order, lag))
    clf = make_pipeline(TangentSpace(metric="riemann"), LogisticRegression(max_iter=500, C=1.0))
    clf.fit(Ctr, ytr)
    probs = np.asarray(clf.predict_proba(Cte), dtype=float)
    pred = probs.argmax(1)
    row = {"fold": str(s), "n": int(len(yte)), "acc": metrics.accuracy(yte, pred),
           "kappa": metrics.kappa(yte, pred), "ece": metrics.ece_from_probs(probs, yte)}
    return row, probs, yte


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--augment", action="store_true", help="time-delay-embed covariances (ACM) before recentering")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--lag", type=int, default=8)
    ap.add_argument("--resample", type=float, default=128.0)
    ap.add_argument("--fmin", type=float, default=8.0)
    ap.add_argument("--fmax", type=float, default=32.0)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel LOSO folds (joblib; -1 = all cores)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true",
                    help="skip updating the committed results.json snapshot (scratch/experimental runs)")
    args = ap.parse_args()

    from joblib import Parallel, delayed

    cfg = EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax)
    meta = store.load(args.dataset, cfg)
    tag = "acm" if args.augment else "ts"
    print(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · recipe {cfg.key()} · recentered {tag}")

    folds = list(splits.leave_one_subject_out(meta))
    print(f"\n=== riemann recentered ({tag}) · cross_subject · {args.dataset} ({len(folds)} folds, jobs={args.jobs}) ===")
    out_folds = Parallel(n_jobs=args.jobs)(
        delayed(_run_fold)(s, tr, te, args.augment, args.order, args.lag) for s, tr, te in folds)

    rows, P, Y = [], [], []
    for row, probs, yte in sorted(out_folds, key=lambda r: r[0]["fold"]):
        rows.append(row); P.append(probs); Y.append(yte)
        print(f"  {row['fold']:>6}  acc {row['acc']:.3f}  kappa {row['kappa']:.3f}  ece {row['ece']:.3f}  (n={row['n']})")

    acc = float(np.mean([r["acc"] for r in rows]))
    kap = float(np.mean([r["kappa"] for r in rows]))
    probs, y = np.concatenate(P), np.concatenate(Y)
    print(f"  {'MEAN':>6}  acc {acc:.3f}  kappa {kap:.3f}")
    print("  vs reference: " + reference.compare(acc, args.dataset, "cross_subject", "riemann"))
    print("  (un-recentered riemann LOSO ~0.357 — the gain is the recentering)")

    out = Path(args.out) if args.out else Path("runs") / f"riemann_recenter_{tag}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    res = {"method": f"riemann_recenter_{tag}", "regime": "cross_subject", "acc_mean": acc,
           "kappa_mean": kap, "per_fold": rows}
    (out / "aggregate.json").write_text(json.dumps(res, indent=2))
    with tracking.run("mindscape", f"riemann_recenter_{tag}_cross",
                      params={"method": f"riemann_recenter_{tag}", "augment": args.augment,
                              "order": args.order, "lag": args.lag},
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
