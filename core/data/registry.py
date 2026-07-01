"""Unified dataset adapter registry — name -> adapter, across modalities (EEG + fNIRS).

The store/splits/harness are modality-agnostic (they ride on the [n,ch,t]+meta schema), so one registry
serves both. Add a dataset = one factory + one `register` line. Adapters are built lazily (heavy imports
— MOABB, scipy.io — deferred to first use), so importing the registry stays cheap.
"""
from __future__ import annotations

from typing import Callable

_FACTORIES: dict[str, Callable] = {}


def register(name: str, factory: Callable) -> None:
    _FACTORIES[name] = factory


def get_adapter(name: str):
    """Adapter for a dataset name (e.g. 'bnci2014_001', 'shin2017_nback')."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def dataset_names() -> list[str]:
    return sorted(_FACTORIES)


# --- registrations (one line per dataset; keep imports inside factories) ---
def _bnci2014_001():
    from core.data.eeg.bnci2014_001 import adapter
    return adapter()


def _shin2017_nback():
    from core.data.fnirs.shin2017 import adapter
    return adapter("nback")


def _shin2017_nback_eeg():
    from core.data.eeg.shin2017_nback_eeg import adapter
    return adapter()


register("bnci2014_001", _bnci2014_001)          # EEG motor imagery
register("shin2017_nback", _shin2017_nback)       # fNIRS n-back workload
register("shin2017_nback_eeg", _shin2017_nback_eeg)  # EEG n-back workload (same task as fNIRS -> Table B)
