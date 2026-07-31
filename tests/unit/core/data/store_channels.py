"""store channel-name persistence — the processed cache is self-describing (one format).

`build` writes a channels.json when the adapter exposes `channels()`; `store.Store.channels()` reads it back, and
returns None for adapters that don't expose names. Uses a tiny fake adapter (no real data / raw files).
"""
import numpy as np
import polars as pl

from core.data import store
from core.data.eeg.base import EpochCfg

CFG = EpochCfg()
NAMES = ["Fp1", "Cz", "POz"]


class _FakeAdapter:
    label_map = {"a": 0, "b": 1}

    def __init__(self, chans=None):
        self._chans = chans

    def subjects(self):
        return [1]

    def get_data(self, subs, cfg):
        X = np.zeros((2, len(NAMES), 4), np.float32)
        return X, np.array([0, 1]), pl.DataFrame({"session": ["0", "0"], "run": ["0", "0"]})

    # `channels` is intentionally added only in the with-names case (via setattr) to exercise the optional path


def _wire(monkeypatch, tmp_path, adapter):
    monkeypatch.setattr(store.Config, "processed_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(store.Registry, "get_adapter", staticmethod(lambda name: adapter))


def test_build_persists_channels_when_adapter_exposes_them(monkeypatch, tmp_path):
    a = _FakeAdapter(NAMES)
    a.channels = lambda: NAMES                                # adapter exposes names
    _wire(monkeypatch, tmp_path, a)
    store.Store.build("fake", CFG)
    assert store.Store.channels("fake", CFG) == NAMES
    assert (store.Store.dataset_dir("fake", CFG) / "channels.json").exists()


def test_channels_is_none_without_adapter_support(monkeypatch, tmp_path):
    _wire(monkeypatch, tmp_path, _FakeAdapter())              # no .channels attribute
    store.Store.build("fake", CFG)
    assert store.Store.channels("fake", CFG) is None
    assert not (store.Store.dataset_dir("fake", CFG) / "channels.json").exists()
