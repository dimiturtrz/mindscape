"""Temperature scaling (Guo 2017) — post-hoc calibration, and the measured finding that it's domain-limited.

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
import logging
from pathlib import Path

import numpy as np
import torch

from core.data import splits, store
from core.data.eeg.base import EpochCfg
from neuroscan import tracking
from neuroscan.evaluation import metrics
from neuroscan.models import decoders

logger = logging.getLogger(__name__)

_EPS = 1e-6              # guard against divide-by-zero when the val fix is ~0 (nothing to transfer)
_TRANSFER_LIMITED = 0.5  # transfer ratio below this: calibration is domain-shift-limited
_TRANSFER_GOOD = 1.2     # transfer ratio at/above this: calibration transfers well cross-session


class TemperatureScaler:
    """Post-hoc temperature scaling (Guo 2017): one scalar T (logits -> logits/T), fit on a held-out val
    set by minimizing NLL with the model frozen. The object OWNS T and the two operations that use it —
    `.fit` sets T, `.ece` reports ECE at the fitted T (or an override). Softmax argmax is unchanged, so
    accuracy is untouched; only confidence (ECE) moves."""

    def __init__(self, T: float = 1.0):
        self.T = T

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        z = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        logT = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logT], lr=0.05, max_iter=80)
        nll = torch.nn.CrossEntropyLoss()

        def closure():                                       # LBFGS requires a closure
            opt.zero_grad()
            loss = nll(z / logT.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(logT.exp().detach())
        return self

    def probs(self, logits: np.ndarray, T: float | None = None) -> np.ndarray:
        """Numerically-stable softmax(logits / T); T defaults to the fitted self.T."""
        z = logits / (self.T if T is None else T)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def ece(self, logits: np.ndarray, labels: np.ndarray, T: float | None = None) -> float:
        p = self.probs(logits, T)
        conf, pred = p.max(1), p.argmax(1)
        return metrics.ece(conf, (pred == labels).astype(float))[0]


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--method", default="atcnet", choices=sorted(decoders.MODELS))
    ap.add_argument("--test-session", default="1test")
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--fmin", type=float, default=4.0)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def _per_subject_rows(meta, fit, test_session):
    """One temperature-scaling row per subject: fit T on the in-session val, report val + cross-session ECE."""
    rows = []
    for s in sorted(meta["subject"].unique().to_list()):
        # train+val from the train session (in-session), test = the eval session (cross-session)
        train, val, test = splits.within_subject(meta, s, test_sessions=[test_session])
        if val.is_empty() or test.is_empty():
            continue
        Xtr, ytr = store.gather(train)
        Xva, yva = store.gather(val)
        Xte, yte = store.gather(test)
        clf = fit(Xtr, ytr)
        lv, lt = clf.predict_logits(Xva), clf.predict_logits(Xte)
        ts = TemperatureScaler().fit(lv, yva)                # fit T on in-session val
        r = {"subject": str(s), "T": round(ts.T, 3),
             "val_ece_uncal": ts.ece(lv, yva, T=1.0), "val_ece_temp": ts.ece(lv, yva),
             "test_ece_uncal": ts.ece(lt, yte, T=1.0), "test_ece_temp": ts.ece(lt, yte),
             "test_acc": metrics.accuracy(yte, lt.argmax(1))}
        rows.append(r)
        logger.info(f"  s{r['subject']}  T {ts.T:.2f} | val ECE {r['val_ece_uncal']:.3f}->{r['val_ece_temp']:.3f} | "
              f"test ECE {r['test_ece_uncal']:.3f}->{r['test_ece_temp']:.3f}  (acc {r['test_acc']:.3f})")
    return rows


def _summarize(rows, method):
    """Aggregate per-subject rows into the summary dict; returns (summary, val_fix, test_fix)."""
    m = {k: float(np.mean([r[k] for r in rows]))
         for k in ("T", "val_ece_uncal", "val_ece_temp", "test_ece_uncal", "test_ece_temp")}
    summary = {"method": method, "regime": "within_calibration", "n": len(rows),
               "T_mean": m["T"],
               "val_ece": {"uncal": m["val_ece_uncal"], "temp": m["val_ece_temp"]},
               "test_ece": {"uncal": m["test_ece_uncal"], "temp": m["test_ece_temp"]},
               "per_subject": rows}
    # the headline read: how much of the val-ECE fix transfers to the cross-session test
    val_fix = summary["val_ece"]["uncal"] - summary["val_ece"]["temp"]
    test_fix = summary["test_ece"]["uncal"] - summary["test_ece"]["temp"]
    summary["transfer_ratio"] = round(test_fix / val_fix, 3) if val_fix > _EPS else None
    return summary, val_fix, test_fix


def _report(summary, method, val_fix, test_fix):
    """Log the val->test ECE transfer and store the verdict on `summary`."""
    logger.info(f"\n=== {method} temperature scaling (in-session val -> cross-session test) ===")
    logger.info(f"  val  ECE {summary['val_ece']['uncal']:.3f} -> {summary['val_ece']['temp']:.3f}  (fixed {val_fix:+.3f})")
    logger.info(f"  test ECE {summary['test_ece']['uncal']:.3f} -> {summary['test_ece']['temp']:.3f}  (fixed {test_fix:+.3f})")
    tr = summary["transfer_ratio"]
    if tr is None:
        verdict = "val already calibrated — nothing to transfer"
    elif tr < _TRANSFER_LIMITED:
        verdict = "calibration is domain-shift-LIMITED (val fix does not transfer to cross-session)"
    elif tr < _TRANSFER_GOOD:
        verdict = "calibration transfers partially across the session shift"
    else:
        verdict = "calibration transfers well (test fixed >= val) — model already low-ECE cross-session"
    logger.info(f"  transfer ratio {tr} — {verdict}")
    summary["verdict"] = verdict


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    args = _parse_args()

    meta = store.load(args.dataset, EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax))
    fit, _ = decoders.make(args.method)

    rows = _per_subject_rows(meta, fit, args.test_session)
    summary, val_fix, test_fix = _summarize(rows, args.method)
    _report(summary, args.method, val_fix, test_fix)

    out = Path(args.out) if args.out else Path("runs") / f"calibrate_{args.method}_{args.dataset}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(summary, indent=2))
    with tracking.run("mindscape", f"calibrate_{args.method}", params={"method": args.method},
                      tags={"method": args.method, "regime": "calibration"}, run_dir=out):
        tracking.metrics({"T_mean": summary["T_mean"],
                          "val_ece_uncal": summary["val_ece"]["uncal"], "val_ece_temp": summary["val_ece"]["temp"],
                          "test_ece_uncal": summary["test_ece"]["uncal"], "test_ece_temp": summary["test_ece"]["temp"]})
        tracking.artifact(out / "calibration.json")
    logger.info(f"-> {out}/calibration.json")


if __name__ == "__main__":
    main()
