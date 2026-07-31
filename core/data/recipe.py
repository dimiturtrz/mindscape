"""The recipe→cache-key contract shared by the EEG and fNIRS preprocessing configs.

`SupportsKey` is the ONE thing the modality-agnostic `Store` needs from a recipe: a cache key. Both
`EpochCfg` (EEG) and `FnirsCfg` (fNIRS) satisfy it structurally, so `Store.load`/`build`/`dataset_dir` take
this contract instead of casting one config through the other at every fNIRS call site. A leaf module (imports
nothing from the data layer), so both modality bases could reference it without an import cycle.
"""
from __future__ import annotations

from typing import Protocol


class SupportsKey(Protocol):
    """A preprocessing recipe that hashes to a filesystem cache key (`processed/<dataset>/<key>/`)."""

    def key(self) -> str: ...
