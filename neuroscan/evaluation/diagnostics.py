"""Stratified diagnostics — break the aggregate number down by the axis that hides the failure.

The siblings stratify by pathology/vendor/defect; mindscape stratifies by **subject** (and session):
the cross-subject spread is exactly where a measured motor-imagery result separates from an inflated one.
"""
from __future__ import annotations

import numpy as np


class Diagnostics:
    @staticmethod
    def spread(rows: list[dict[str, object]], key: str = "acc") -> dict[str, float]:
        """Summary of a per-group metric: mean / std / min / max — the cross-subject variability that the
        single mean number hides."""
        vals = np.array([r[key] for r in rows], float)
        return {"mean": float(vals.mean()), "std": float(vals.std()),
                "min": float(vals.min()), "max": float(vals.max())}
