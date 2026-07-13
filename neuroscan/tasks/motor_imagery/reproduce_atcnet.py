"""Reproduce published ATCNet (~0.81) on BCI IV-2a via braindecode-canonical preprocessing.

Proves the pipeline can reproduce the literature before we trust our own contribution numbers. Uses
continuous-signal EMS (core/data/eeg/braindecode_pre), full-trial ATCNet (its own internal window aug),
no extra standardization, their recipe (500 epochs, no early stop). Per-subject hold-out: session-1
train -> session-2 test.

    python -m neuroscan.tasks.motor_imagery.reproduce_atcnet --epochs 500
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl

from core.data.eeg import braindecode_pre
from core.data.eeg.braindecode_pre import BraindecodePreConfig
from neuroscan import tracking
from neuroscan.evaluation import metrics
from neuroscan.models import decoders

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="atcnet", choices=sorted(decoders.MODELS))
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=0, help="0 = no early stop (their recipe)")
    ap.add_argument("--seeds", type=int, default=1, help="runs per subject to average (published = 10-run mean)")
    ap.add_argument("--standardize", default="zscore", choices=["zscore", "none"],
                    help="zscore = StandardScaler (the published ATCNet recipe); none = continuous EMS")
    ap.add_argument("--batch", type=int, default=64, help="batch size (Altaheri uses 64)")
    ap.add_argument("--out", default="runs/reproduce")
    args = ap.parse_args()

    # zscore: bandpassed uV from preprocessing + trainer z-score (== Altaheri StandardScaler).
    # none:   continuous EMS applied in preprocessing, trainer passes through.
    use_ems = args.standardize == "none"
    logger.info(f"preprocessing ({'continuous EMS' if use_ems else 'bandpass uV + z-score'}) {args.method} ...")
    X, y, meta = braindecode_pre.BraindecodePre.get_data("BNCI2014_001", subjects=args.subjects,
                                          config=BraindecodePreConfig(ems=use_ems))
    logger.info(f"X {X.shape} · sessions {sorted(meta['session'].unique().to_list())}")

    fit, _ = decoders.make(args.method)
    rows, models = [], []
    for s in sorted(meta["subject"].unique().to_list()):
        idx = (meta["subject"] == s).to_numpy()
        Xs, ys = X[idx], y[idx]
        sess = meta.filter(pl.col("subject") == s)["session"].to_numpy()
        a, b = sorted(set(sess.tolist()))                 # train session, test session
        tr, te = sess == a, sess == b
        accs, kaps = [], []
        for seed in range(args.seeds):
            clf = fit(Xs[tr], ys[tr], standardize=args.standardize, crop_frac=None, batch=args.batch,
                      epochs=args.epochs, patience=args.patience, log_every=0, seed=seed)
            pred = clf.predict_proba(Xs[te]).argmax(1)
            accs.append(metrics.accuracy(ys[te], pred))
            kaps.append(metrics.kappa(ys[te], pred))
        models.append((str(s), clf))                      # keep the last-seed model per subject to persist
        r = {"subject": str(s), "acc": float(np.mean(accs)), "kappa": float(np.mean(kaps)),
             "acc_std": float(np.std(accs)), "seeds": args.seeds, "n": int(te.sum())}
        rows.append(r)
        spread = f"  (per-seed {min(accs):.3f}-{max(accs):.3f})" if args.seeds > 1 else ""
        logger.info(f"  s{r['subject']}  acc {r['acc']:.3f}  kappa {r['kappa']:.3f}{spread}")

    acc = float(np.mean([r["acc"] for r in rows]))
    kap = float(np.mean([r["kappa"] for r in rows]))
    logger.info(f"\n=== {args.method} reproduction (braindecode EMS, {args.epochs}ep) ===")
    logger.info(f"  MEAN acc {acc:.3f}  kappa {kap:.3f}   (published ~0.81 / 0.76)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"method": args.method, "epochs": args.epochs, "standardize": args.standardize,
              "batch": args.batch, "seeds": args.seeds, "acc_mean": acc, "kappa_mean": kap,
              "per_subject": rows}
    (out / f"{args.method}.json").write_text(json.dumps(report, indent=2))
    with tracking.run("mindscape", f"reproduce_{args.method}",
                      params={"method": args.method, "standardize": args.standardize,
                              "batch": args.batch, "epochs": args.epochs, "seeds": args.seeds},
                      tags={"kind": "reproduction"}, run_dir=out):
        tracking.metrics({"acc_mean": acc, "kappa_mean": kap})
        tracking.per_group("acc_subject", {r["subject"]: r["acc"] for r in rows})
        tracking.artifact(out / f"{args.method}.json")
        for subj, clf in models:                          # persist the trained net per subject
            tracking.save_model(clf, f"model_{args.method}_s{subj}", run_dir=out)
    logger.info(f"-> {out}/{args.method}.json")


if __name__ == "__main__":
    main()
