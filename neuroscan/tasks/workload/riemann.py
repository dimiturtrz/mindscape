"""Shared Riemannian cross-subject decode for the workload source/fusion probes.

OAS spatial covariance -> per-subject re-centered tangent space -> LR
(`transfer.zero_shot_predict`) — the method that gives the workload EEG cross-subject reference.
"""
from __future__ import annotations

from collections.abc import Iterator

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
    def cross_subject_folds(c: Float[np.ndarray, "n ch ch"], y: Int[np.ndarray, "n"], g: Int[np.ndarray, "n"],
                            seeds: list[int], k: int
                            ) -> Iterator[tuple[Int[np.ndarray, "m"], Float[np.ndarray, "m cls"]]]:
        """`seeds` x `k`-fold grouped by subject: per fold, per-subject re-centered tangent + LR, yields
        `(y_test, test_proba)`. The one shared cross-subject fold loop — callers own their own aggregation
        (accuracy / kappa / confusion) off the per-fold predictions."""
        for seed in seeds:
            for tr, te in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(c, y, g):
                proba = transfer.zero_shot_predict(transfer.Domain(c[tr], y[tr], g[tr]),
                                                   transfer.Domain(c[te], groups=g[te]), scale=False)
                yield y[te], proba

    @staticmethod
    def cross_subject_decode(c: Float[np.ndarray, "n ch ch"], y: Int[np.ndarray, "n"], g: Int[np.ndarray, "n"],
                             seeds: list[int], k: int) -> tuple[float, float]:
        """`seeds` x `k`-fold grouped by subject: re-centered tangent + LR, returns (mean, std) accuracy."""
        accs = [metrics.Metrics.accuracy(yt, proba.argmax(1))
                for yt, proba in Riemann.cross_subject_folds(c, y, g, seeds, k)]
        return float(np.mean(accs)), float(np.std(accs))
