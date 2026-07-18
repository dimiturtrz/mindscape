"""Run-end invariant checks for a retrieval result — catch a plausible-but-wrong number AT run time.

A concurrent-CUDA crash once ate ~1h of GPU here; the cheaper waste is a run that finishes with a silently
inconsistent number (a CI point that isn't the reported mean, a bracket that excludes its own point, a
top-1 that landed below chance because labels were misaligned). These are the pennies-cheap asserts that
catch exactly that the instant `evaluate` returns, so a broken run fails loud instead of feeding a table.

Loud by default (one WARNING per violation); `strict=True` raises — for tests / a CI smoke. Advisory, never
a gate: the numbers are the science, this only guards their internal consistency.
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

_TOL = 1e-6


class Invariants:
    """Assert a retrieval run's reported numbers are internally consistent (the guardrail against silent-wrong).

    `res` is the `evaluate`/`train` dict: `single_trial` {k: acc}, `single_trial_ci` {k: (point, lo, hi)},
    and (from `train`) `chance_top1`. Keys may be ints (live) or strings (post-JSON) — both are accepted."""

    @staticmethod
    def check(res: dict[str, Any], *, strict: bool = False) -> list[str]:
        """-> list of violation messages (empty = clean). Logs each loudly; raises AssertionError if `strict`."""
        violations = (Invariants._bounded(res) + Invariants._brackets(res)
                      + Invariants._ci_matches_mean(res) + Invariants._above_chance(res))
        for m in violations:
            logger.warning(f"INVARIANT VIOLATION: {m}")
        if strict and violations:
            raise AssertionError("; ".join(violations))
        return violations

    @staticmethod
    def _finite(*xs: float) -> bool:
        return all(math.isfinite(x) for x in xs)               # NaN/inf -> skip the undefined check

    @staticmethod
    def _acc(res: dict[str, Any]) -> dict[int, float]:
        return {int(k): float(v) for k, v in res.get("single_trial", {}).items()}

    @staticmethod
    def _ci(res: dict[str, Any]) -> dict[int, tuple[float, float, float]]:
        return {int(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in res.get("single_trial_ci", {}).items()}

    @staticmethod
    def _bounded(res: dict[str, Any]) -> list[str]:
        """Every reported accuracy must be a finite probability in [0, 1]."""
        return [f"single-trial top{k} = {a} not a finite [0,1] probability"
                for k, a in Invariants._acc(res).items() if not (Invariants._finite(a) and 0.0 <= a <= 1.0)]

    @staticmethod
    def _brackets(res: dict[str, Any]) -> list[str]:
        """Every bootstrap bracket must contain its own point estimate."""
        return [f"top{k} bracket [{lo:.4f}, {hi:.4f}] excludes point {p:.4f}"
                for k, (p, lo, hi) in Invariants._ci(res).items()
                if Invariants._finite(p, lo, hi) and not lo <= p <= hi]

    @staticmethod
    def _ci_matches_mean(res: dict[str, Any]) -> list[str]:
        """The CI point and the reported mean are the SAME statistic — they must agree (catches a hits-vector
        vs headline mismatch: the bootstrap resampling something other than what the table reports)."""
        acc = Invariants._acc(res)
        return [f"top{k} ci-point {p:.4f} != reported mean {acc[k]:.4f} (aggregate mismatch)"
                for k, (p, _lo, _hi) in Invariants._ci(res).items()
                if k in acc and Invariants._finite(p, acc[k]) and abs(p - acc[k]) > _TOL]

    @staticmethod
    def _above_chance(res: dict[str, Any]) -> list[str]:
        """A finished run scoring BELOW chance on top-1 is almost always a broken run (label misalignment,
        wrong candidate bank) — not a real result. Advisory, but loud: chance is the sanity floor."""
        acc, chance = Invariants._acc(res), res.get("chance_top1")
        top1 = acc.get(1)
        if chance is None or top1 is None or not Invariants._finite(top1, chance):
            return []
        return [f"top1 {top1:.4f} below chance {chance:.4f} — likely a broken run"] if top1 < chance else []

    @staticmethod
    def reconciles(delta: float, point_a: float, point_b: float, label: str = "delta") -> bool:
        """A paired delta must equal point_b − point_a (same statistic) — the s1t2 guard so a mislabeled
        arm fails immediately. Loud + returns ok/bad."""
        ok = Invariants._finite(delta, point_a, point_b) and abs(delta - (point_b - point_a)) <= _TOL
        if not ok:
            logger.warning(f"INVARIANT VIOLATION: {label} {delta:.4f} != {point_b:.4f} - {point_a:.4f}")
        return ok
