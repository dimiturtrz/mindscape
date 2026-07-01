"""Per-subject z-score (the EEG-workload transfer lever) — the pure normalization, no data/model.

Each subject is standardized by ITS OWN feature stats (removing the subject-specific offset that sinks
cross-subject band-power).
"""
import numpy as np

from neuroscan.tasks.workload.calibration_ablation import _zscore


def _two_subjects():
    # subject A: offset +10, scale 2; subject B: offset -5, scale 0.5 — different location AND scale
    rng = np.random.default_rng(0)
    A = 10 + 2 * rng.normal(size=(30, 3))
    B = -5 + 0.5 * rng.normal(size=(30, 3))
    F = np.vstack([A, B])
    g = np.array(["A"] * 30 + ["B"] * 30)
    return F, g


def test_zscore_standardizes_each_subject_independently():
    F, g = _two_subjects()
    Z = _zscore(F, g)
    for s in ("A", "B"):
        m = g == s
        assert np.allclose(Z[m].mean(0), 0.0, atol=1e-9)     # each subject centred
        assert np.allclose(Z[m].std(0), 1.0, atol=1e-6)      # ...and unit-scaled by its OWN stats


def test_zscore_removes_between_subject_offset():
    F, g = _two_subjects()
    raw_gap = abs(F[g == "A"].mean() - F[g == "B"].mean())
    z = _zscore(F, g)
    z_gap = abs(z[g == "A"].mean() - z[g == "B"].mean())
    assert raw_gap > 10 and z_gap < 1e-6                     # the subject offset is gone
