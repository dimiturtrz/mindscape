"""One data root, everything derived — mirrors the siblings' core/config.py.

Source of truth: `paths.yaml` at the repo root (gitignored). Copy `paths.example.yaml` ->
`paths.yaml` and set the single `data:` line. Override with the env var MINDSCAPE_DATA (e.g. in CI).

Under <data> the layout is:
    <data>/raw/        MOABB/MNE download cache (you don't touch it; MOABB fills it)
    <data>/processed/  epoched preprocess cache (created on first run)

We also point MOABB/MNE at <data>/raw so downloads land inside the one root, not ~/mne_data.
"""
from __future__ import annotations

import os
from pathlib import Path

from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parent.parent


def data_root(sub: str | None = None) -> Path:
    """The single data root, or a named subdir under it (`raw` / `processed`)."""
    env = os.environ.get("MINDSCAPE_DATA")
    if env:
        root = Path(env)
    else:
        cfg = _REPO / "paths.yaml"
        if not cfg.exists():
            raise FileNotFoundError(
                f"{cfg} not found — copy paths.example.yaml -> paths.yaml and set `data:` "
                f"(or set the MINDSCAPE_DATA env var)."
            )
        root = Path(OmegaConf.load(cfg).data)
    return root / sub if sub else root


def raw_dir() -> Path:
    return data_root("raw")


def processed_dir() -> Path:
    return data_root("processed")


def configure_moabb_download() -> Path:
    """Point MOABB/MNE's download cache at <data>/raw so recordings stay inside the one root.
    Idempotent; returns the cache dir. Call before any MOABB dataset access."""
    cache = raw_dir()
    cache.mkdir(parents=True, exist_ok=True)
    # MNE reads MNE_DATA; MOABB reads MNE_DATA for most datasets.
    os.environ.setdefault("MNE_DATA", str(cache))
    os.environ.setdefault("MOABB_RESULTS", str(processed_dir() / "moabb_results"))
    return cache
