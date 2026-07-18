"""Unified dataset adapter registry — name -> adapter, across modalities (EEG + fNIRS).

The store/splits/harness are modality-agnostic (they ride on the [n,ch,t]+meta schema), so one registry
serves both. Add a dataset = one factory + one `register` line. Adapters are built lazily (the factory is
only called on `get_adapter`), so no dataset is instantiated until requested.
"""
from __future__ import annotations

from typing import Callable

from core.data.eeg.base import DatasetAdapter
from core.data.eeg.bnci2014_001 import Bnci2014001
from core.data.eeg.shin2017_nback_eeg import Shin2017NbackEegAdapter
from core.data.fnirs.shin2017 import Shin2017NirsAdapter

_FACTORIES: dict[str, Callable[[], DatasetAdapter]] = {}


class Registry:
    """Unified dataset adapter registry — the free helpers folded in as staticmethods (public names kept).
    The built-in registrations run lazily on first `get_adapter` (kept out of import time — no side effects
    on import), so importing this module instantiates nothing."""

    _populated = False

    @staticmethod
    def register(name: str, factory: Callable[[], DatasetAdapter]) -> None:
        _FACTORIES[name] = factory

    @staticmethod
    def get_adapter(name: str) -> DatasetAdapter:
        """Adapter for a dataset name (e.g. 'bnci2014_001', 'shin2017_nback')."""
        Registry._ensure_populated()
        if name not in _FACTORIES:
            raise KeyError(f"unknown dataset {name!r}; have {sorted(_FACTORIES)}")
        return _FACTORIES[name]()

    @staticmethod
    def _ensure_populated() -> None:
        """Register the built-in dataset factories exactly once, on first use (not at import)."""
        if Registry._populated:
            return
        Registry._populated = True
        Registry.register("bnci2014_001", Registry._bnci2014_001)              # EEG motor imagery
        Registry.register("shin2017_nback", Registry._shin2017_nback)          # fNIRS n-back workload
        Registry.register("shin2017_nback_eeg", Registry._shin2017_nback_eeg)  # EEG n-back workload (-> Table B)

    # --- built-in dataset factories (one per dataset) ---
    @staticmethod
    def _bnci2014_001():
        return Bnci2014001.adapter()

    @staticmethod
    def _shin2017_nback():
        return Shin2017NirsAdapter.adapter("nback")

    @staticmethod
    def _shin2017_nback_eeg():
        return Shin2017NbackEegAdapter.adapter()
