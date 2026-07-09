"""Unified dataset adapter registry — name -> adapter, across modalities (EEG + fNIRS).

The store/splits/harness are modality-agnostic (they ride on the [n,ch,t]+meta schema), so one registry
serves both. Add a dataset = one factory + one `register` line. Adapters are built lazily (the factory is
only called on `get_adapter`), so no dataset is instantiated until requested.
"""
from __future__ import annotations

from typing import Callable

from core.data.eeg.base import DatasetAdapter
from core.data.eeg.bnci2014_001 import adapter as _bnci2014_001_adapter
from core.data.eeg.shin2017_nback_eeg import adapter as _shin2017_nback_eeg_adapter
from core.data.fnirs.shin2017 import adapter as _shin2017_nback_adapter

_FACTORIES: dict[str, Callable] = {}


def register(name: str, factory: Callable) -> None:
    _FACTORIES[name] = factory


def get_adapter(name: str) -> DatasetAdapter:
    """Adapter for a dataset name (e.g. 'bnci2014_001', 'shin2017_nback')."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


# --- registrations (one line per dataset) ---
def _bnci2014_001():
    return _bnci2014_001_adapter()


def _shin2017_nback():
    return _shin2017_nback_adapter("nback")


def _shin2017_nback_eeg():
    return _shin2017_nback_eeg_adapter()


register("bnci2014_001", _bnci2014_001)          # EEG motor imagery
register("shin2017_nback", _shin2017_nback)       # fNIRS n-back workload
register("shin2017_nback_eeg", _shin2017_nback_eeg)  # EEG n-back workload (same task as fNIRS -> Table B)
