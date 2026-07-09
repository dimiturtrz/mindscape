"""Source-localization pure helpers (bd 728) — cache-key + montage validation, no fsaverage/forward build.

The fsaverage forward/inverse (`build_inverse`, `to_parcels`) needs template data + MNE modeling and is
`# pragma: no cover`; here we pin the config-hashing and the montage guard that gate it.
"""
import pytest

from core.features.eeg.source import SourceConfig, _cache_key, _montage_info


def test_cache_key_deterministic_and_input_sensitive():
    cfg = SourceConfig()
    chs = ["C3", "Cz", "C4"]
    assert _cache_key(chs, 250.0, cfg) == _cache_key(chs, 250.0, cfg)          # deterministic
    assert _cache_key(chs, 250.0, cfg) != _cache_key(chs, 128.0, cfg)          # sfreq matters
    assert _cache_key(chs, 250.0, cfg) != _cache_key(["C3", "C4"], 250.0, cfg)  # montage matters
    assert _cache_key(chs, 250.0, cfg) != _cache_key(chs, 250.0, SourceConfig(spacing="oct6"))  # cfg matters


def test_montage_info_rejects_unknown_channels():
    with pytest.raises(ValueError, match="not in standard_1005"):
        _montage_info(["C3", "NOTACHAN"], 250.0)


def test_montage_info_builds_average_referenced_eeg_info():
    info = _montage_info(["C3", "Cz", "C4"], 250.0)
    assert info["sfreq"] == 250.0 and len(info["ch_names"]) == 3
    assert len(info["projs"]) == 1                                             # average-reference projection added
