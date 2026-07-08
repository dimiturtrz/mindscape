"""Shared EEG dataset primitives + the DatasetAdapter interface.

Canonical motor-imagery label convention (fixed across datasets, so one decoder's classes mean the
same thing everywhere — the EEG analogue of the siblings' canonical mask labels):

    0 left_hand   1 right_hand   2 feet   3 tongue

Every adapter remaps its source event names to this via `label_map`. An epoch tensor is
[n_epochs, n_channels, n_times] float32; labels are canonical int ids; metadata is one row per epoch
(subject, session, run) — the frame splits filter on (see data/splits.py).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import polars as pl
from pydantic import BaseModel

CANONICAL_MI: dict[str, int] = {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
CANONICAL_MI_NAMES: dict[int, str] = {v: k for k, v in CANONICAL_MI.items()}


class EpochCfg(BaseModel):
    """Preprocessing params that define an epoched cache. Two recipes never collide (see `key`).

    Defaults = a standard motor-imagery recipe: 8–32 Hz band (mu+beta), resampled to 128 Hz, the
    cue-locked window the paradigm provides (tmax=None -> paradigm default)."""
    fmin: float = 8.0
    fmax: float = 32.0
    tmin: float = 0.0
    tmax: float | None = None
    resample: float = 128.0

    def key(self) -> str:
        """Cache key encoding the recipe -> processed/<dataset>/<key>/."""
        def f(x):
            return str(x).replace(".", "p").replace("-", "m")
        tmax = "full" if self.tmax is None else f(self.tmax)
        return f"b{f(self.fmin)}-{f(self.fmax)}_t{f(self.tmin)}-{tmax}_r{f(self.resample)}"


@runtime_checkable
class DatasetAdapter(Protocol):
    """What every dataset adapter provides — the dataset-agnostic interface the pipeline rides on.

    `label_map` makes the per-dataset event convention *interface data*, not hardcoded logic.
    `get_data` returns canonical epochs for the requested subjects (all if None)."""
    name: str
    n_classes: int
    label_map: dict[str, int]   # source event name -> canonical id

    def subjects(self) -> list[int]: ...
    def get_data(self, subjects: list[int] | None, cfg: EpochCfg
                 ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]: ...


class MoabbMIAdapter:
    """Reusable adapter over a MOABB motor-imagery dataset + the MotorImagery paradigm.

    Most MOABB MI sets share this exact shape, so a concrete dataset (bnci2014_001.py, physionet_mi.py)
    is just: pick the MOABB dataset class, the canonical class count, and the label map. New dataset =
    one file + one registry line.
    """

    def __init__(self, name: str, dataset_cls, n_classes: int = 4,
                 label_map: dict[str, int] | None = None):
        self.name = name
        self._dataset_cls = dataset_cls
        self.n_classes = n_classes
        # default: MOABB already standardizes MI event strings to the canonical names
        self.label_map = label_map or {k: v for k, v in CANONICAL_MI.items()}

    def _dataset(self):
        return self._dataset_cls()

    def subjects(self) -> list[int]:
        return list(self._dataset().subject_list)

    def get_data(self, subjects: list[int] | None, cfg: EpochCfg
                 ) -> tuple[np.ndarray, np.ndarray, pl.DataFrame]:
        """Epoch the requested subjects -> (X[n,ch,t] float32, y[n] canonical int, meta polars frame).

        meta columns: subject, session, run (one row per epoch). Pulls downloads into <data>/raw."""
        from moabb.paradigms import MotorImagery

        from core.config import configure_moabb_download
        configure_moabb_download()

        ds = self._dataset()
        paradigm = MotorImagery(n_classes=self.n_classes, fmin=cfg.fmin, fmax=cfg.fmax,
                                tmin=cfg.tmin, tmax=cfg.tmax, resample=cfg.resample)
        X, labels, meta = paradigm.get_data(dataset=ds, subjects=subjects or self.subjects())
        y = np.array([self.label_map[str(label)] for label in labels], dtype=np.int64)
        m = pl.from_pandas(meta[["subject", "session", "run"]].astype(str))
        return X.astype(np.float32), y, m
