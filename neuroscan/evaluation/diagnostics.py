"""Stratified diagnostics — break the aggregate number down by the axis that hides the failure.

The siblings stratify by pathology/vendor/defect; mindscape stratifies by **subject** (and session):
the cross-subject spread is exactly where a measured motor-imagery result separates from an inflated one.
"""
from __future__ import annotations

import numpy as np

from neuroscan.evaluation import metrics


def by_group(group: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
             probs: np.ndarray | None = None) -> list[dict]:
    """Per-group {group, n, acc, kappa, ece} rows, sorted by group. `group` is one label per sample
    (e.g. subject id or session). ECE included when `probs` given."""
    group = np.asarray(group)
    rows = []
    for g in sorted(set(group.tolist())):
        m = group == g
        row = {"group": str(g), "n": int(m.sum()),
               "acc": metrics.accuracy(y_true[m], y_pred[m]),
               "kappa": metrics.kappa(y_true[m], y_pred[m])}
        if probs is not None:
            row["ece"] = metrics.ece_from_probs(probs[m], y_true[m])
        rows.append(row)
    return rows


def spread(rows: list[dict], key: str = "acc") -> dict:
    """Summary of a per-group metric: mean / std / min / max — the cross-subject variability that the
    single mean number hides."""
    vals = np.array([r[key] for r in rows], float)
    return {"mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max())}
