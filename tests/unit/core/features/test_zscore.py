"""Per-subject z-score (`zscore_per_subject`) — the unsupervised cross-subject normalization lever.

Each subject is standardized by ITS OWN feature stats: within-subject the features become ~0 mean / unit
std, a held-out subject standardizes by its own stats (no train leakage), and the between-subject offset
that sinks cross-subject band-power is removed. Pure math — synthetic features, no data/model.
"""
import numpy as np

from core.features import zscore_per_subject


def _two_subjects():
    # subject A: offset +10, scale 2; subject B: offset -5, scale 0.5 — different LOCATION and SCALE
    rng = np.random.default_rng(0)
    A = 10 + 2 * rng.normal(size=(30, 3))
    B = -5 + 0.5 * rng.normal(size=(30, 3))
    F = np.vstack([A, B])
    g = np.array(["A"] * 30 + ["B"] * 30)
    return F, g


def test_each_subject_standardized_by_its_own_stats():
    F, g = _two_subjects()
    Z = zscore_per_subject(F, g)
    for s in ("A", "B"):
        m = g == s
        assert np.allclose(Z[m].mean(0), 0.0, atol=1e-9)     # each subject centred
        assert np.allclose(Z[m].std(0), 1.0, atol=1e-6)      # ...and unit-scaled by its OWN stats


def test_removes_between_subject_offset():
    F, g = _two_subjects()
    raw_gap = abs(F[g == "A"].mean() - F[g == "B"].mean())
    Z = zscore_per_subject(F, g)
    z_gap = abs(Z[g == "A"].mean() - Z[g == "B"].mean())
    assert raw_gap > 10 and z_gap < 1e-6                     # the subject offset is gone


def test_held_out_subject_uses_its_own_stats_not_others():
    """Unsupervised: a subject's normalization depends only on ITS rows. Adding a second subject must not
    change the first subject's z-scored output (no pooling, no cross-subject leakage)."""
    rng = np.random.default_rng(1)
    A = 3 + 1.5 * rng.normal(size=(20, 4))
    B = -7 + 4 * rng.normal(size=(20, 4))
    ga = np.array(["A"] * 20)
    Za_alone = zscore_per_subject(A, ga)
    Z_joint = zscore_per_subject(np.vstack([A, B]), np.array(["A"] * 20 + ["B"] * 20))
    assert np.allclose(Za_alone, Z_joint[:20])              # A unchanged by B's presence


def test_zero_variance_feature_is_finite():
    """A constant feature within a subject (std 0) is guarded by the +1e-6 -> finite, ~0, not NaN/inf."""
    F = np.column_stack([np.full(10, 5.0), np.arange(10, dtype=float)])
    Z = zscore_per_subject(F, np.zeros(10, int))
    assert np.isfinite(Z).all()
    assert np.allclose(Z[:, 0], 0.0)                        # constant column collapses to 0, not blows up
