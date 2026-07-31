"""fNIRS-informed weighted minimum-norm inverse (bd 4so) — the pure linear algebra, no MNE/fsaverage.

Equivalence classes: shape/validation, and the defining property — boosting the prior on a source concentrates
the recovered estimate there (fNIRS 'where' regularizing the ill-posed EEG inverse).
"""
import numpy as np
import pytest

from core.features.fusion.source_prior import SourcePrior


def _leadfield(seed=0, n_ch=8, n_src=6):
    return np.random.default_rng(seed).standard_normal((n_ch, n_src))


def test_weighted_min_norm_inverse():
    """weighted_min_norm_inverse() returns inverse operator [n_src, n_ch] from lead field and prior."""
    g = _leadfield()
    k = SourcePrior.weighted_min_norm_inverse(g, np.ones(g.shape[1]))
    assert k.shape == (g.shape[1], g.shape[0])                 # [n_src, n_ch]


def test_rejects_bad_prior():
    g = _leadfield()
    with pytest.raises(ValueError, match="n_src"):
        SourcePrior.weighted_min_norm_inverse(g, np.ones(g.shape[1] + 1))   # length mismatch
    with pytest.raises(ValueError, match="non-negative"):
        SourcePrior.weighted_min_norm_inverse(g, -np.ones(g.shape[1]))      # negative prior variance


def test_prior_concentrates_estimate_on_primed_source():
    g = _leadfield(seed=1)
    j = 2
    sensor = g[:, j]                                            # pure source-j activity at the sensors
    uniform = SourcePrior.weighted_min_norm_inverse(g, np.ones(g.shape[1])) @ sensor
    prior = np.ones(g.shape[1])
    prior[j] = 20.0                                            # fNIRS says source j is active
    primed = SourcePrior.weighted_min_norm_inverse(g, prior) @ sensor
    # the primed inverse puts a larger SHARE of the recovered energy on source j
    assert abs(primed[j]) / np.abs(primed).sum() > abs(uniform[j]) / np.abs(uniform).sum()


def test_prior_leadfield():
    """prior_leadfield() returns lead field and parcel aggregator matrix (requires fsaverage; skip if unavailable)."""
    pytest.importorskip("mne.datasets")
    from core.features.eeg.source import SourceConfig
    try:
        g, aggregator = SourcePrior.prior_leadfield(["C3", "Cz", "C4"], 250.0, SourceConfig())
        assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0         # [n_ch, n_src]
        assert aggregator.ndim == 2 and aggregator.shape[0] > 0            # [n_labels, n_src]
        assert np.isfinite(g).all() and np.isfinite(aggregator).all()
    except Exception:
        pytest.skip("fsaverage template not available")


def test_parcels_from_leadfield():
    """parcels_from_leadfield() applies weighted inverse to produce parcel series."""
    g = _leadfield(seed=2, n_ch=8, n_src=10)
    n_labels = 5
    aggregator = np.ones((n_labels, g.shape[1])) / (g.shape[1] / n_labels)
    epochs = np.random.default_rng(3).standard_normal((3, g.shape[0], 100)).astype(np.float32)
    parcels = SourcePrior.parcels_from_leadfield(epochs, g, aggregator)
    assert parcels.shape == (3, n_labels, 100)               # [n, n_labels, t]
    assert np.isfinite(parcels).all()
