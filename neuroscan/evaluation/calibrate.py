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
from typing import Callable, TypedDict

import numpy as np
import polars as pl

from core.data import splits, store
from core.data.eeg.base import EpochCfg
from neuroscan import tracking
from neuroscan.evaluation import metrics
from neuroscan.evaluation.temperature_scaler import TemperatureScaler
from neuroscan.models import decoders
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)

_EPS = 1e-6              # guard against divide-by-zero when the val fix is ~0 (nothing to transfer)
_TRANSFER_LIMITED = 0.5  # transfer ratio below this: calibration is domain-shift-limited
_TRANSFER_GOOD = 1.2     # transfer ratio at/above this: calibration transfers well cross-session


class EcePair(TypedDict):
    """Uncalibrated vs temperature-scaled ECE, the pair reported for each of val / test."""
    uncal: float
    temp: float


class CalibSummary(TypedDict):
    """The calibration run's aggregate record (written to calibration.json)."""
    method: str
    regime: str
    n: int
    T_mean: float
    val_ece: EcePair
    test_ece: EcePair
    per_subject: list[dict[str, str | float]]
    transfer_ratio: float | None
    verdict: str


class Calibrate:
    @classmethod
    def _parse_args(cls):
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--dataset", default="bnci2014_001")
        ap.add_argument("--method", default="atcnet", choices=sorted(decoders.MODELS))
        ap.add_argument("--test-session", default="1test")
        ap.add_argument("--resample", type=float, default=250.0)
        ap.add_argument("--fmin", type=float, default=4.0)
        ap.add_argument("--fmax", type=float, default=40.0)
        ap.add_argument("--out", default=None)
        return ap.parse_args()

    @classmethod
    def _per_subject_rows(cls, meta: pl.DataFrame, fit: Callable[..., object],
                          test_session: str) -> list[dict[str, str | float]]:
        """One temperature-scaling row per subject: fit T on the in-session val, report val + cross-session ECE."""
        rows: list[dict[str, str | float]] = []
        for s in sorted(meta["subject"].unique().to_list()):
            # train+val from the train session (in-session), test = the eval session (cross-session)
            train, val, test = splits.Splits.within_subject(meta, s, test_sessions=(test_session,))
            if val.is_empty() or test.is_empty():
                continue
            Xtr, ytr = store.Store.gather(train)
            Xva, yva = store.Store.gather(val)
            Xte, yte = store.Store.gather(test)
            clf = fit(Xtr, ytr)
            lv, lt = clf.predict_logits(Xva), clf.predict_logits(Xte)
            ts = TemperatureScaler().fit(lv, yva)                # fit T on in-session val
            r = {"subject": str(s), "T": round(ts.T, 3),
                 "val_ece_uncal": ts.ece(lv, yva, T=1.0), "val_ece_temp": ts.ece(lv, yva),
                 "test_ece_uncal": ts.ece(lt, yte, T=1.0), "test_ece_temp": ts.ece(lt, yte),
                 "test_acc": metrics.Metrics.accuracy(yte, lt.argmax(1))}
            rows.append(r)
            logger.info(f"  s{r['subject']}  T {ts.T:.2f} | "
                  f"val ECE {r['val_ece_uncal']:.3f}->{r['val_ece_temp']:.3f} | "
                  f"test ECE {r['test_ece_uncal']:.3f}->{r['test_ece_temp']:.3f}  (acc {r['test_acc']:.3f})")
        return rows

    @classmethod
    def _summarize(cls, rows: list[dict[str, str | float]], method: str) -> tuple[CalibSummary, float, float]:
        """Aggregate per-subject rows into the summary dict; returns (summary, val_fix, test_fix)."""
        m = {k: float(np.mean([float(r[k]) for r in rows]))
             for k in ("T", "val_ece_uncal", "val_ece_temp", "test_ece_uncal", "test_ece_temp")}
        # the headline read: how much of the val-ECE fix transfers to the cross-session test
        val_fix = m["val_ece_uncal"] - m["val_ece_temp"]
        test_fix = m["test_ece_uncal"] - m["test_ece_temp"]
        summary: CalibSummary = {
            "method": method, "regime": "within_calibration", "n": len(rows), "T_mean": m["T"],
            "val_ece": {"uncal": m["val_ece_uncal"], "temp": m["val_ece_temp"]},
            "test_ece": {"uncal": m["test_ece_uncal"], "temp": m["test_ece_temp"]},
            "per_subject": rows,
            "transfer_ratio": round(test_fix / val_fix, 3) if val_fix > _EPS else None,
            "verdict": "",
        }
        return summary, val_fix, test_fix

    @classmethod
    def _report(cls, summary: CalibSummary, method: str, val_fix: float, test_fix: float) -> None:
        """Log the val->test ECE transfer and store the verdict on `summary`."""
        logger.info(f"\n=== {method} temperature scaling (in-session val -> cross-session test) ===")
        logger.info(f"  val  ECE {summary['val_ece']['uncal']:.3f} -> "
                    f"{summary['val_ece']['temp']:.3f}  (fixed {val_fix:+.3f})")
        logger.info(f"  test ECE {summary['test_ece']['uncal']:.3f} -> "
                    f"{summary['test_ece']['temp']:.3f}  (fixed {test_fix:+.3f})")
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

    @classmethod
    def main(cls):
        Cli.setup_logging()
        args = cls._parse_args()

        meta = store.Store.load(args.dataset, EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax))
        fit, _ = decoders.BraindecodeClf.make(args.method)

        rows = cls._per_subject_rows(meta, fit, args.test_session)
        summary, val_fix, test_fix = cls._summarize(rows, args.method)
        cls._report(summary, args.method, val_fix, test_fix)

        out = Path(args.out) if args.out else Path("runs") / f"calibrate_{args.method}_{args.dataset}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "calibration.json").write_text(json.dumps(summary, indent=2))
        with tracking.Tracking.run("mindscape", f"calibrate_{args.method}", params={"method": args.method},
                          tags={"method": args.method, "regime": "calibration"}, run_dir=out):
            tracking.Tracking.metrics({
                "T_mean": summary["T_mean"],
                "val_ece_uncal": summary["val_ece"]["uncal"], "val_ece_temp": summary["val_ece"]["temp"],
                "test_ece_uncal": summary["test_ece"]["uncal"], "test_ece_temp": summary["test_ece"]["temp"],
            })
            tracking.Tracking.artifact(out / "calibration.json")
        logger.info(f"-> {out}/calibration.json")


if __name__ == "__main__":
    Calibrate.main()
