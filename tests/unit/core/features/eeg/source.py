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
