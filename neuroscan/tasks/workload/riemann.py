"""Shared Riemannian cross-subject decode for the workload source/fusion probes.

OAS spatial covariance -> per-subject re-centered tangent space -> LR
(`transfer.zero_shot_predict`) — the method that gives the workload EEG cross-subject reference.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int
from pyriemann.estimation import Covariances
from sklearn.model_selection import StratifiedGroupKFold

from baselines.eeg import transfer
from neuroscan.evaluation import metrics


class Riemann:
    """OAS covariance + per-subject re-centered tangent + LR cross-subject decode."""

    @staticmethod
    def cov(x: Float[np.ndarray, "n ch t"]) -> Float[np.ndarray, "n ch ch"]:
        return Covariances("oas").transform(x.astype(np.float64))

    @staticmethod
    def cross_subject_decode(c: Float[np.ndarray, "n ch ch"], y: Int[np.ndarray, "n"], g: Int[np.ndarray, "n"],
                             seeds: list[int], k: int) -> tuple[float, float]:
        """`seeds` x `k`-fold grouped by subject: re-centered tangent + LR, returns (mean, std) accuracy."""
        accs = []
        for seed in seeds:
            for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(c, y, g):
                proba = transfer.zero_shot_predict(transfer.Domain(c[tr], y[tr], g[tr]),
                                                   transfer.Domain(c[te], groups=g[te]), scale=False)
                accs.append(metrics.Metrics.accuracy(y[te], proba.argmax(1)))
        return float(np.mean(accs)), float(np.std(accs))
