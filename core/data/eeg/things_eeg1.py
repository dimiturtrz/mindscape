"""THINGS-EEG1 (Grootswagers et al. 2022, OpenNeuro ds003825) — the second EEG->image dataset.

Shares the THINGS concept set with THINGS-EEG2, so the two enable a cross-DATASET zero-shot retrieval test
(train the encoder on one, retrieve on the other). Unlike EEG2, the trial->concept mapping is carried inline
in the BIDS `events.tsv` (`object` = the THINGS concept NAME), so identity is the name — which is exactly the
bridge to EEG2 (its concepts are a subset of EEG1's 1,854). See research/deep_dives for the full layout.

Raw: 63-ch BrainVision @ 1000 Hz, 10 Hz RSVP (SOA 100 ms), one session/run per subject, 50 subjects. We epoch
off the raw with OUR preprocessing (per-channel z-score, optional band-pass, resample) for EEG2 parity, so the
representation is comparable across datasets.

    <data>/raw/things_eeg1/sub-01/eeg/sub-01_task-rsvp_{eeg.vhdr,events.tsv}

Channel count is 63 (verified off a real subject), matching EEG2 — so the cross-dataset test has no channel-
count mismatch to reconcile (montage/reference alignment is still a D-layer concern). Other discrepancies:
10 Hz single-shot (low per-image SNR) vs EEG2's 5 Hz with test-set repeats; image-file identity is concept-level.
"""
from __future__ import annotations

import logging

import mne
import numpy as np
import polars as pl
from jaxtyping import Float
from pydantic import BaseModel
from scipy.signal import resample as _resample

from core.config import Config
from core.data.signal import Signal

logger = logging.getLogger(__name__)

_ROOT = "things_eeg1"
_N_EEG = 63            # all channels are EEG (verified) — no trailing stim channel; events come from events.tsv
_FS_RAW = 1000.0
# events.tsv columns we rely on (BIDS + the dataset's own extension): stimulus onset, THINGS concept name,
# image filename, and the two trial flags used to subset (held-out validation stimuli / fixation-colour target).
_COL_ONSET, _COL_CONCEPT, _COL_FILE = "onset", "object", "stimname"
_COL_ISTEST, _COL_ISTARGET = "isteststim", "istarget"


class ThingsEeg1EpochCfg(BaseModel):
    """Epoching knobs, matched to ThingsEpochCfg (EEG2) so the two representations line up. `include_validation`
    keeps the 200 held-out validation stimuli (default: main set only); `drop_targets` excludes the fixation-
    colour target trials (they carry a concept but are a different task)."""
    tmin: float = 0.0
    tmax: float = 1.0
    resample: float = 250.0
    fmin: float | None = None
    fmax: float | None = None
    include_validation: bool = False
    drop_targets: bool = True


class ThingsEeg1:
    """THINGS-EEG1 on-disk index + own-preprocessing epoching — the free helpers folded in as staticmethods
    (public names kept). The second EEG->image dataset, matched to EEG2 for the cross-dataset retrieval test."""

    @staticmethod
    def _subject_dir(subject: int):
        return Config.raw_dir() / _ROOT / f"sub-{subject:02d}"

    @staticmethod
    def subjects() -> list[int]:
        """Downloaded subject ids (sub-NN dirs present under the raw tree), sorted."""
        root = Config.raw_dir() / _ROOT
        if not root.is_dir():
            return []
        return sorted(int(path.name[4:]) for path in root.glob("sub-*") if path.name[4:].isdigit())

    @staticmethod
    def channels() -> list[str]:
        """The 63 EEG channel names (10-10 montage), read from a subject's BrainVision header — constant across
        subjects. Used to montage-align against THINGS-EEG2 for the cross-dataset test (they share 62 electrodes,
        EEG1 has Fz where EEG2 has Cz)."""
        subject = ThingsEeg1.subjects()[0]
        raw = mne.io.read_raw_brainvision(
            ThingsEeg1._subject_dir(subject) / "eeg" / f"sub-{subject:02d}_task-rsvp_eeg.vhdr",
            preload=False, verbose="ERROR")
        return list(raw.ch_names[:_N_EEG])

    @staticmethod
    def _row_mask(events: pl.DataFrame, cfg: ThingsEeg1EpochCfg) -> np.ndarray:
        """Which events.tsv rows to epoch: drop fixation-colour targets and (by default) the held-out validation
        stimuli. Flags are optional columns — absent means the dataset doesn't distinguish that split, keep all."""
        mask = np.ones(len(events), dtype=bool)
        if cfg.drop_targets and _COL_ISTARGET in events.columns:
            mask &= (events[_COL_ISTARGET].to_numpy() == 0)
        if not cfg.include_validation and _COL_ISTEST in events.columns:
            mask &= (events[_COL_ISTEST].to_numpy() == 0)
        return mask

    @staticmethod
    def epochs_from_events(eeg: Float[np.ndarray, "ch t"], fs: float, events: pl.DataFrame, cfg: ThingsEeg1EpochCfg
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pure epoching: continuous `eeg` [ch, T] + a BIDS `events` frame -> (epochs [n, ch, t], concept_name[n],
        img_file[n]). Onset is BIDS seconds -> sample @ fs. Windows overrunning the recording are dropped. Per-
        channel z-score + optional band-pass/resample mirror the EEG2 adapter so the two datasets are comparable.
        No file IO here (the BrainVision read is the caller's job), so the windowing/filtering path is testable.
        """
        if cfg.fmin is not None or cfg.fmax is not None:
            eeg = Signal.bandpass(eeg, cfg.fmin or 0.1, cfg.fmax or (fs / 2 - 1), fs)

        rows = events.filter(pl.Series(ThingsEeg1._row_mask(events, cfg)))
        onset = np.rint(rows[_COL_ONSET].to_numpy() * fs).astype(int)      # BIDS onset is SECONDS
        if len(onset) and onset.max() >= eeg.shape[1]:
            raise ValueError(f"onset sample {onset.max()} exceeds recording length {eeg.shape[1]} — "
                             "check the events.tsv onset units (BIDS = seconds)")

        start, stop = round(cfg.tmin * fs), round(cfg.tmax * fs)
        keep = (onset + start >= 0) & (onset + stop <= eeg.shape[1])
        onset = onset[keep]
        rows = rows.filter(pl.Series(keep))

        epochs = np.stack([eeg[:, at + start:at + stop] for at in onset]).astype(np.float32)   # [n, ch, t]
        # per-channel z-score: EEG is in volts (~1e-5); O(1) scaling keeps the encoder's BatchNorm conditioned
        # (the same fix as EEG2 — without it eval-mode embeddings collapse).
        epochs = (epochs - epochs.mean(axis=2, keepdims=True)) / (epochs.std(axis=2, keepdims=True) + 1e-7)
        if cfg.resample and cfg.resample != fs:
            epochs = _resample(epochs, round(epochs.shape[2] * cfg.resample / fs), axis=2).astype(np.float32)
        return epochs, rows[_COL_CONCEPT].to_numpy(), rows[_COL_FILE].to_numpy()

    @staticmethod
    def _subject_epochs(subject: int, cfg: ThingsEeg1EpochCfg):
        """Read one subject's BrainVision raw + events.tsv, then epoch (the IO shell around epochs_from_events)."""
        eeg_dir = ThingsEeg1._subject_dir(subject) / "eeg"
        raw = mne.io.read_raw_brainvision(
            eeg_dir / f"sub-{subject:02d}_task-rsvp_eeg.vhdr", preload=True, verbose="ERROR")
        events = pl.read_csv(eeg_dir / f"sub-{subject:02d}_task-rsvp_events.tsv", separator="\t")
        return ThingsEeg1.epochs_from_events(raw.get_data()[:_N_EEG], raw.info["sfreq"], events, cfg)

    @staticmethod
    def get_epochs(subjects_: list[int] | None = None, config: ThingsEeg1EpochCfg | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pl.DataFrame]:
        """Epoch THINGS-EEG1 for the EEG->image task off the raw. Returns (eeg [n,63,t] float32, concept_name [n]
        str, img_file [n] str, meta {subject}). concept_name is the THINGS name — the cross-dataset bridge to EEG2
        (map name -> the shared CLIP target the same way clip_targets does)."""
        cfg = config or ThingsEeg1EpochCfg()
        chosen = subjects_ or ThingsEeg1.subjects()
        eeg_parts, concept_parts, file_parts, subject_col = [], [], [], []
        for subject in chosen:
            eeg, concept, files = ThingsEeg1._subject_epochs(subject, cfg)
            eeg_parts.append(eeg)
            concept_parts.append(concept)
            file_parts.append(files)
            subject_col += [str(subject)] * len(concept)
            logger.info(f"sub-{subject:02d}: {len(concept)} epochs {eeg.shape[1:]}")
        return (np.concatenate(eeg_parts).astype(np.float32),
                np.concatenate(concept_parts), np.concatenate(file_parts),
                pl.DataFrame({"subject": subject_col}))
