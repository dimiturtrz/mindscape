"""Reproduce published ATCNet (~0.81) on BCI IV-2a via braindecode-canonical preprocessing.

Proves the pipeline can reproduce the literature before we trust our own contribution numbers. Uses
continuous-signal EMS (core/data/eeg/braindecode_pre), full-trial ATCNet (its own internal window aug),
no extra standardization, their recipe (500 epochs, no early stop). Per-subject hold-out: session-1
train -> session-2 test.

    python -m neuroscan.experiments.reproduce_atcnet --epochs 500
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from core.data.eeg import braindecode_pre
from neuroscan.evaluation import metrics
from neuroscan.models import decoders


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="atcnet", choices=sorted(decoders.MODELS))
    ap.add_argument("--subjects", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--patience", type=int, default=0, help="0 = no early stop (their recipe)")
    ap.add_argument("--seeds", type=int, default=1, help="runs per subject to average (published = 10-run mean)")
    ap.add_argument("--out", default="runs/reproduce")
    args = ap.parse_args()

    print(f"preprocessing (continuous EMS) {args.method} ...")
    X, y, meta = braindecode_pre.get_data("BNCI2014_001", subjects=args.subjects)
    print(f"X {X.shape} · sessions {sorted(meta['session'].unique().to_list())}")

    fit, _ = decoders.make(args.method)
    rows = []
    for s in sorted(meta["subject"].unique().to_list()):
        idx = (meta["subject"] == s).to_numpy()
        Xs, ys = X[idx], y[idx]
        sess = meta.filter(pl.col("subject") == s)["session"].to_numpy()
        a, b = sorted(set(sess.tolist()))                 # train session, test session
        tr, te = sess == a, sess == b
        accs, kaps = [], []
        for seed in range(args.seeds):
            clf = fit(Xs[tr], ys[tr], standardize="none", crop_frac=None,
                      epochs=args.epochs, patience=args.patience, log_every=0, seed=seed)
            pred = clf.predict_proba(Xs[te]).argmax(1)
            accs.append(metrics.accuracy(ys[te], pred))
            kaps.append(metrics.kappa(ys[te], pred))
        r = {"subject": str(s), "acc": float(np.mean(accs)), "kappa": float(np.mean(kaps)),
             "acc_std": float(np.std(accs)), "seeds": args.seeds, "n": int(te.sum())}
        rows.append(r)
        spread = f"  (per-seed {min(accs):.3f}-{max(accs):.3f})" if args.seeds > 1 else ""
        print(f"  s{r['subject']}  acc {r['acc']:.3f}  kappa {r['kappa']:.3f}{spread}")

    acc = float(np.mean([r["acc"] for r in rows]))
    kap = float(np.mean([r["kappa"] for r in rows]))
    print(f"\n=== {args.method} reproduction (braindecode EMS, {args.epochs}ep) ===")
    print(f"  MEAN acc {acc:.3f}  kappa {kap:.3f}   (published ~0.81 / 0.76)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.method}.json").write_text(json.dumps(
        {"method": args.method, "epochs": args.epochs, "acc_mean": acc, "kappa_mean": kap,
         "per_subject": rows}, indent=2))
    print(f"-> {out}/{args.method}.json")


if __name__ == "__main__":
    main()
