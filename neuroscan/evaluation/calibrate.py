"""Temperature scaling (Guo 2017) — post-hoc calibration, and the honest finding that it's domain-limited.

Fit a single scalar T (logits -> logits/T) on a held-out IN-SESSION val set by minimizing NLL, model
frozen; T>1 softens overconfidence. It does NOT change argmax, so accuracy is untouched — only the
confidence (ECE) moves.

The point of interest is the *transfer*: T calibrated on in-session val typically fixes the val ECE but
NOT the cross-session (eval-session) test ECE — i.e. post-hoc calibration is itself domain-shift-limited.
This is the EEG echo of the siblings' cross-vendor calibration finding.

    python -m neuroscan.evaluation.calibrate --method atcnet --resample 250 --fmin 4 --fmax 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from core.data import store, splits
from core.data.eeg.base import EpochCfg
from neuroscan import tracking
from neuroscan.evaluation import metrics
from neuroscan.models import decoders


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """T>0 minimizing NLL of softmax(logits/T) vs labels (LBFGS on log T). Model frozen; one scalar."""
    import torch

    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=80)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(z / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp().detach())


def ece_at(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    """ECE of softmax(logits/T) vs labels."""
    z = logits / T
    z = z - z.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    return metrics.ece(conf, (pred == labels).astype(float))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--method", default="atcnet", choices=sorted(decoders.MODELS))
    ap.add_argument("--test-session", default="1test")
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--fmin", type=float, default=4.0)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = store.load(args.dataset, EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax))
    fit, _ = decoders.make(args.method)

    rows = []
    for s in sorted(meta["subject"].unique().to_list()):
        # train+val from the train session (in-session), test = the eval session (cross-session)
        train, val, test = splits.within_subject(meta, s, test_sessions=[args.test_session])
        if val.is_empty() or test.is_empty():
            continue
        Xtr, ytr = store.gather(train)
        Xva, yva = store.gather(val)
        Xte, yte = store.gather(test)
        clf = fit(Xtr, ytr)
        lv, lt = clf.predict_logits(Xva), clf.predict_logits(Xte)
        T = fit_temperature(lv, yva)
        r = {"subject": str(s), "T": round(T, 3),
             "val_ece_uncal": ece_at(lv, yva, 1.0), "val_ece_temp": ece_at(lv, yva, T),
             "test_ece_uncal": ece_at(lt, yte, 1.0), "test_ece_temp": ece_at(lt, yte, T),
             "test_acc": metrics.accuracy(yte, lt.argmax(1))}
        rows.append(r)
        print(f"  s{r['subject']}  T {T:.2f} | val ECE {r['val_ece_uncal']:.3f}->{r['val_ece_temp']:.3f} | "
              f"test ECE {r['test_ece_uncal']:.3f}->{r['test_ece_temp']:.3f}  (acc {r['test_acc']:.3f})")

    def mean(k):
        return float(np.mean([r[k] for r in rows]))

    summary = {"method": args.method, "regime": "within_calibration", "n": len(rows),
               "T_mean": mean("T"),
               "val_ece": {"uncal": mean("val_ece_uncal"), "temp": mean("val_ece_temp")},
               "test_ece": {"uncal": mean("test_ece_uncal"), "temp": mean("test_ece_temp")},
               "per_subject": rows}
    # the headline read: how much of the val-ECE fix transfers to the cross-session test
    val_fix = summary["val_ece"]["uncal"] - summary["val_ece"]["temp"]
    test_fix = summary["test_ece"]["uncal"] - summary["test_ece"]["temp"]
    summary["transfer_ratio"] = round(test_fix / val_fix, 3) if val_fix > 1e-6 else None

    print(f"\n=== {args.method} temperature scaling (in-session val -> cross-session test) ===")
    print(f"  val  ECE {summary['val_ece']['uncal']:.3f} -> {summary['val_ece']['temp']:.3f}  (fixed {val_fix:+.3f})")
    print(f"  test ECE {summary['test_ece']['uncal']:.3f} -> {summary['test_ece']['temp']:.3f}  (fixed {test_fix:+.3f})")
    tr = summary["transfer_ratio"]
    if tr is None:
        verdict = "val already calibrated — nothing to transfer"
    elif tr < 0.5:
        verdict = "calibration is domain-shift-LIMITED (val fix does not transfer to cross-session)"
    elif tr < 1.2:
        verdict = "calibration transfers partially across the session shift"
    else:
        verdict = "calibration transfers well (test fixed >= val) — model already low-ECE cross-session"
    print(f"  transfer ratio {tr} — {verdict}")
    summary["verdict"] = verdict

    out = Path(args.out) if args.out else Path("runs") / f"calibrate_{args.method}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(summary, indent=2))
    with tracking.run("mindscape", f"calibrate_{args.method}", params={"method": args.method},
                      tags={"method": args.method, "regime": "calibration"}, run_dir=out):
        tracking.metrics({"T_mean": summary["T_mean"],
                          "val_ece_uncal": summary["val_ece"]["uncal"], "val_ece_temp": summary["val_ece"]["temp"],
                          "test_ece_uncal": summary["test_ece"]["uncal"], "test_ece_temp": summary["test_ece"]["temp"]})
        tracking.artifact(out / "calibration.json")
    print(f"-> {out}/calibration.json")


if __name__ == "__main__":
    main()
