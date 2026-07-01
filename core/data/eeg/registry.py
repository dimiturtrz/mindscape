"""Back-compat shim — the registry moved up to `core.data.registry` (unified across EEG + fNIRS).

Kept so existing `from core.data.eeg.registry import get_adapter` imports still resolve.
"""
from __future__ import annotations

from core.data.registry import dataset_names, get_adapter, register  # noqa: F401
