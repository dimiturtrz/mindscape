"""Braindecode-canonical preprocessing: continuous-signal EMS done RIGHT (before epoching), then window.

This is the correct exponential-moving-standardization path — running stats stabilize over the whole
recording, not per-epoch (the per-epoch version regressed; see research/2026-06-30_2a_sota_recipe.md).
Returns (X, y, meta) in mindscape's schema so it feeds the same harness/decoders.

Recipe (braindecode 2a example / Schirrmeister 2017): pick EEG -> V to uV -> bandpass 4-38 Hz ->
exponential_moving_standardize(factor_new=1e-3, init_block_size=1000) on the CONTINUOUS signal ->
windows from events with a -0.5 s trial-start offset.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    exponential_moving_standardize,
    preprocess,
)
from pydantic import BaseModel

from core.config import Config


class BraindecodePreConfig(BaseModel):
    """The braindecode-canonical preprocessing knobs. `fmin`/`fmax` = bandpass; `trial_start_offset_s` = the
    window's start offset relative to the trial cue; `factor_new`/`init_block_size` = the EMS running-stat
    parameters; `ems` toggles continuous exponential-moving standardization."""
    fmin: float = 4.0
    fmax: float = 38.0
    trial_start_offset_s: float = -0.5
    factor_new: float = 1e-3
    init_block_size: int = 1000
    ems: bool = True


class BraindecodePre:
    """Braindecode-canonical preprocessing (continuous EMS then window) as a staticmethod (public name kept) —
    the correct exponential-moving-standardization path, returning mindscape-schema (X, y, meta)."""

    @staticmethod
    def get_data(dataset_name: str = "BNCI2014_001", subjects: list[int] | None = None,
                 config: BraindecodePreConfig | None = None):
        """Return (X [n,ch,t] float32, y [n] int, meta polars{subject,session,run}).

        `config.ems=True` applies continuous exponential-moving standardization here (braindecode default);
        `ems=False` returns bandpassed microvolts only — use with the trainer's z-score (StandardScaler),
        which is what the published ATCNet pipeline actually does."""
        cfg = config or BraindecodePreConfig()
        Config.configure_moabb_download()

        ds = MOABBDataset(dataset_name=dataset_name, subject_ids=subjects)
        sfreq = ds.datasets[0].raw.info["sfreq"]
        steps = [
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            Preprocessor(lambda d: d * 1e6),                                  # V -> microvolts
            Preprocessor("filter", l_freq=cfg.fmin, h_freq=cfg.fmax),
        ]
        if cfg.ems:
            steps.append(Preprocessor(exponential_moving_standardize, factor_new=cfg.factor_new,
                                      init_block_size=cfg.init_block_size))
        preprocess(ds, steps)

        start = round(cfg.trial_start_offset_s * sfreq)
        windows = create_windows_from_events(ds, trial_start_offset_samples=start,
                                             trial_stop_offset_samples=0, preload=True)

        Xs, ys, subj, sess, run = [], [], [], [], []
        for wds in windows.datasets:
            d = wds.description
            arr = np.stack([wds[i][0] for i in range(len(wds))]).astype(np.float32)   # [n, ch, t]
            lab = np.array([wds[i][1] for i in range(len(wds))], dtype=np.int64)        # braindecode int labels
            Xs.append(arr)
            ys.append(lab)
            n = len(lab)
            subj += [str(d["subject"])] * n
            sess += [str(d["session"])] * n
            run += [str(d.get("run", "0"))] * n

        X = np.concatenate(Xs).astype(np.float32)
        y = np.concatenate(ys).astype(np.int64)
        meta = pl.DataFrame({"subject": subj, "session": sess, "run": run})
        return X, y, meta
