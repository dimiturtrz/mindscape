"""BCI Competition IV-2a (MOABB `BNCI2014_001`) — the Stage-0 warm-up dataset.

9 subjects · 4 classes (left/right hand, feet, tongue) · 22 EEG ch @ 250 Hz · 2 sessions × 288 trials.
The most-used motor-imagery benchmark; the published within-subject ceiling (~70–75% for the standard
CSP/EEGNet methods, ~88% for transformer SOTA) is what the harness quarantines against.
"""
from __future__ import annotations

from moabb.datasets import BNCI2014_001

from core.data.eeg.base import MoabbMIAdapter


class Bnci2014001:
    """BCI Competition IV-2a adapter factory (public name kept) — the Stage-0 motor-imagery warm-up set."""

    @staticmethod
    def adapter() -> MoabbMIAdapter:
        return MoabbMIAdapter(name="bnci2014_001", dataset_cls=BNCI2014_001, n_classes=4)
