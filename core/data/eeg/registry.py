"""Dataset adapter registry — name -> adapter. Add a dataset = one file + one line here.

Adapters are built lazily (the MOABB import + dataset object is deferred) so importing the registry
is cheap and doesn't require MOABB until a dataset is actually requested.
"""
from __future__ import annotations

from core.data.eeg.base import DatasetAdapter

# name -> zero-arg factory returning the adapter
_FACTORIES: dict[str, callable] = {}


def register(name: str, factory) -> None:
    _FACTORIES[name] = factory


def get_adapter(name: str) -> DatasetAdapter:
    """Adapter for a dataset name (e.g. 'bnci2014_001')."""
    if name not in _FACTORIES:
        raise KeyError(f"unknown dataset {name!r}; have {sorted(_FACTORIES)}")
    return _FACTORIES[name]()


def dataset_names() -> list[str]:
    return sorted(_FACTORIES)


# --- registrations (one line per dataset) ---
def _bnci2014_001():
    from core.data.eeg.bnci2014_001 import adapter
    return adapter()


register("bnci2014_001", _bnci2014_001)
