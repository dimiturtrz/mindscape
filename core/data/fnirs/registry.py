"""Back-compat / symmetry shim — datasets register in the unified `core.data.registry`.

Mirrors core/data/eeg/registry.py so both modalities expose the same import surface.
"""
from __future__ import annotations

from core.data.registry import get_adapter, register  # noqa: F401
