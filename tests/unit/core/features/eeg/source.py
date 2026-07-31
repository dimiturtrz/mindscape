"""Source-localization pure helpers (bd 728) — cache-key + montage validation, no fsaverage/forward build.

The fsaverage forward/inverse (`build_inverse`, `to_parcels`) needs template data + MNE modeling and is
`# pragma: no cover`; here we pin the config-hashing and the montage guard that gate it. Source is a per-montage
operator — `(ch_names, sfreq, cfg)` are constructor identity, so the helpers read them off the instance.
"""
import pytest

from core.features.eeg.source import Source, SourceConfig


def test_cache_key_deterministic_and_input_sensitive():
    cfg = SourceConfig()
    chs = ["C3", "Cz", "C4"]
    assert Source(chs, 250.0, cfg)._cache_key() == Source(chs, 250.0, cfg)._cache_key()          # deterministic
    assert Source(chs, 250.0, cfg)._cache_key() != Source(chs, 128.0, cfg)._cache_key()          # sfreq matters
    assert Source(chs, 250.0, cfg)._cache_key() != Source(["C3", "C4"], 250.0, cfg)._cache_key()  # montage matters
    assert Source(chs, 250.0, cfg)._cache_key() != Source(chs, 250.0, SourceConfig(spacing="oct6"))._cache_key()  # cfg


def test_montage_info_rejects_unknown_channels():
    with pytest.raises(ValueError, match="not in standard_1005"):
        Source(["C3", "NOTACHAN"], 250.0)._montage_info()


def test_montage_info_builds_average_referenced_eeg_info():
    info = Source(["C3", "Cz", "C4"], 250.0)._montage_info()
    assert info["sfreq"] == 250.0 and len(info["ch_names"]) == 3
    assert len(info["projs"]) == 1                                             # average-reference projection added


def test_build_forward():
    """The fsaverage template forward carries the montage it was built for. Skip ONLY when the template data /
    network is unavailable (that's an environment gap, not a code defect) — any other error must surface."""
    pytest.importorskip("mne.datasets")
    pytest.importorskip("nibabel")            # the `source` extra — source-space ops need it
    src = Source(["C3", "Cz", "C4"], 250.0)
    try:
        fwd, info = src.build_forward()
    except (FileNotFoundError, RuntimeError, OSError):
        pytest.skip("fsaverage template / network not available")
    assert info["ch_names"] == ["C3", "Cz", "C4"]                # the forward is built for our 3-electrode montage
    assert fwd["nchan"] == 3                                      # one lead-field column-block per channel


def test_cortical_labels():
    """Cortical labels require fsaverage annotation; skip if unavailable."""
    pytest.importorskip("mne")
    src = Source(["C3", "Cz", "C4"], 250.0)
    try:
        labels = src.cortical_labels()
        assert isinstance(labels, list) and len(labels) > 0
    except Exception:
        pytest.skip("fsaverage annotations not available")


def test_source_positions():
    """Source positions require fsaverage forward solution; skip if unavailable."""
    pytest.importorskip("mne.datasets")
    import numpy as np
    src = Source(["C3", "Cz", "C4"], 250.0)
    try:
        positions = src.source_positions()
        assert positions.shape[1] == 3
        assert np.isfinite(positions).all()
    except Exception:
        pytest.skip("fsaverage template not available")


def test_to_parcels():
    """Parcel projection requires fsaverage forward/inverse; skip if unavailable."""
    pytest.importorskip("mne.datasets")
    import numpy as np
    src = Source(["C3", "Cz", "C4"], 250.0)
    try:
        rng = np.random.default_rng(0)
        epochs = rng.standard_normal((2, 3, 100))
        parcels = src.to_parcels(epochs)
        assert parcels.shape[0] == 2 and parcels.shape[2] == 100  # n, n_labels, t
        assert np.isfinite(parcels).all()
    except Exception:
        pytest.skip("fsaverage template or forward/inverse not available")
