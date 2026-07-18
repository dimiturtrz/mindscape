"""Paired bootstrap delta-CI between two retrieval runs — 830 (does ft beat NICE?) WITHOUT folds.

The deferred LOSO question — is CBraMod-ft's ~2.3% single-trial top-1 really above NICE's ~1.6%, or is the
~0.7pp gap scatter? — answered honestly from ONE run each, no multi-fold retrain (the owner's anti-fold /
stage-gated-rigor rule). Both runs are scored on the SAME test subject's SAME trials in the SAME order, so
the per-trial hit vectors (persisted by `train --out`, bd 5s3l) are paired: a paired bootstrap resamples
shared trial indices, cancels the shared test-set noise, and puts a CI on the delta. CI excludes 0 -> real
lift; straddles 0 -> the gap is noise.

    python -m neuroscan.tasks.visual.paired_delta --a nice_test5.json --b cbramod_ft_test5.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from jaxtyping import Float

from neuroscan.evaluation.invariants import Invariants
from neuroscan.evaluation.metrics import BootCfg, Metrics
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)


class PairedDelta:
    """Paired bootstrap delta-CI between two persisted retrieval runs (their per-trial hit vectors)."""

    @classmethod
    def _hits(cls, result: dict[str, Any], k: int) -> Float[np.ndarray, "n"]:
        """The per-trial 0/1 hit vector at top-k from a `train --out` result (JSON keys are strings)."""
        return np.asarray(result["single_trial_hits"][str(k)], dtype=float)

    @classmethod
    def compare(cls, hits_a: dict[int, np.ndarray], hits_b: dict[int, np.ndarray],
                cfg: BootCfg) -> dict[int, dict[str, Any]]:
        """Per-k: CI on each arm + the PAIRED delta CI (b − a). Requires the two arms scored on the same
        trials in the same order (asserts equal length). Returns {k: {a, b, delta}} of (point, lo, hi)."""
        out: dict[int, dict[str, Any]] = {}
        for k in sorted(hits_a):
            a, b = hits_a[k], hits_b[k]
            if len(a) != len(b):
                raise ValueError(f"top{k}: arms have {len(a)} vs {len(b)} trials — not paired")
            ci_a = Metrics.boot_ci(np.mean, a, cfg=cfg)
            ci_b = Metrics.boot_ci(np.mean, b, cfg=cfg)
            delta = Metrics.boot_delta_ci(np.mean, [a], [b], cfg=cfg)
            Invariants.reconciles(delta[0], ci_a[0], ci_b[0], label=f"top{k} delta")
            out[k] = {"a": ci_a, "b": ci_b, "delta": delta}
        return out

    @classmethod
    def _fmt(cls, ci: tuple[float, float, float]) -> str:
        p, lo, hi = (100 * x for x in ci)
        return f"{p:.2f}% [{lo:.2f}, {hi:.2f}]"

    @classmethod
    def report(cls, name_a: str, name_b: str, comparison: dict[int, dict[str, Any]]) -> None:
        for k, r in comparison.items():
            verdict = "REAL (CI excludes 0)" if r["delta"][1] > 0 else \
                      "noise (CI straddles 0)" if r["delta"][2] >= 0 else "REVERSED"
            logger.info(f"top{k}:  {name_a} {cls._fmt(r['a'])}  |  {name_b} {cls._fmt(r['b'])}")
            logger.info(f"        delta (b-a) {cls._fmt(r['delta'])}  ->  {verdict}")

    @classmethod
    def main(cls):
        Cli.setup_logging()
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--a", required=True, help="run A result json (baseline, e.g. NICE) with single_trial_hits")
        ap.add_argument("--b", required=True, help="run B result json (candidate, e.g. CBraMod-ft)")
        ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap resamples")
        ap.add_argument("--alpha", type=float, default=0.05, help="two-sided level (0.05 -> 95% CI)")
        args = ap.parse_args()

        res_a = json.loads(Path(args.a).read_text())
        res_b = json.loads(Path(args.b).read_text())
        ks = [int(k) for k in res_a["single_trial_hits"]]
        hits_a = {k: cls._hits(res_a, k) for k in ks}
        hits_b = {k: cls._hits(res_b, k) for k in ks}
        comparison = cls.compare(hits_a, hits_b, BootCfg(n_boot=args.n_boot, alpha=args.alpha))
        cls.report(Path(args.a).stem, Path(args.b).stem, comparison)


if __name__ == "__main__":
    PairedDelta.main()
