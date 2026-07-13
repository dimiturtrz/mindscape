"""EEG → cortical source space (bd 728) — the lead-field + inverse operator that moves sensor signals onto
the cortex, so fusion and decoding can work where the physics lives (source space), not on the volume-conducted
sensor mixture.

Template pipeline (no individual MRI): the FreeSurfer **fsaverage** head + a standard-montage forward solution
(lead field) + a minimum-norm inverse (dSPM, ad-hoc noise cov). Sensor epochs `[n, ch, t]` project to a
**Desikan-Killiany parcellation** `[n, 68, t]` — a compact, anatomically-named source representation. The forward
solution is the expensive step (~30 s per montage), so the built inverse operator is cached to disk keyed by
(montage, sfreq, config).

This is a foundation operator, not a decode result: it gives the source-space substrate the Stage-4 fusion needs
(EEG cortical estimate to co-register with the fNIRS optical sources). `build_inverse` / `to_parcels`.
"""
from __future__ import annotations

import hashlib
import logging
import os

import mne
import numpy as np
from pydantic import BaseModel

from core.config import Config

logger = logging.getLogger(__name__)


class SourceConfig(BaseModel):
    """Template source-localization knobs. Defaults = a standard dSPM template pipeline."""
    spacing: str = "oct5"          # source-space resolution (oct5 ≈ 1026 src/hemi)
    method: str = "dSPM"           # minimum-norm inverse variant
    snr: float = 3.0               # regularization: lambda2 = 1 / snr²
    parcellation: str = "aparc"    # Desikan-Killiany (68 cortical labels)
    label_mode: str = "mean_flip"  # sign-flip-aware label aggregation (cancels within-label cancellation)


class Source:
    """EEG → cortical source space (bd 728) — the lead-field + inverse operator for ONE montage. The montage
    `(ch_names, sfreq)` + `cfg` ARE the operator's identity (they define the forward/inverse and key its disk
    cache), so they live in `__init__`, not threaded through every call. `build_inverse` / `to_parcels` are
    the entry points."""

    def __init__(self, ch_names: list[str], sfreq: float, cfg: SourceConfig | None = None):
        self.ch_names = ch_names
        self.sfreq = sfreq
        self.cfg = cfg or SourceConfig()

    @staticmethod
    def _fsaverage_dir() -> tuple[str, str]:   # pragma: no cover — needs the fsaverage template data
        """(fsaverage path, subjects_dir), fetched/cached by MNE under the data root."""
        fs = mne.datasets.fetch_fsaverage(verbose=False)
        return fs, os.path.dirname(fs)

    def _montage_info(self):
        """An average-referenced EEG `Info` with the channels placed on the standard 10-05 montage."""
        montage = mne.channels.make_standard_montage("standard_1005")
        known = set(montage.ch_names)
        missing = [c for c in self.ch_names if c not in known]
        if missing:
            raise ValueError(f"channels not in standard_1005 montage: {missing}")
        info = mne.create_info(list(self.ch_names), float(self.sfreq), "eeg")
        info.set_montage(montage)
        raw = mne.io.RawArray(np.zeros((len(self.ch_names), 1)), info, verbose=False)
        return raw.set_eeg_reference("average", projection=True, verbose=False).info

    def _cache_key(self) -> str:
        payload = "|".join(self.ch_names) + f"|{self.sfreq}|{self.cfg.spacing}|{self.cfg.method}|{self.cfg.snr}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def build_forward(self):  # pragma: no cover — needs fsaverage template data
        """The fsaverage template forward solution (lead field) + its `Info` for the montage. The expensive step —
        shared by the dSPM inverse (`build_inverse`, 728) and the fNIRS-informed weighted inverse (4so)."""
        fs, subjects_dir = Source._fsaverage_dir()
        info = self._montage_info()
        src = mne.setup_source_space("fsaverage", spacing=self.cfg.spacing, subjects_dir=subjects_dir,
                                     add_dist=False, verbose=False)
        bem = os.path.join(fs, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
        fwd = mne.make_forward_solution(info, trans="fsaverage", src=src, bem=bem, eeg=True, meg=False,
                                        verbose=False)
        return fwd, info

    def cortical_labels(self):  # pragma: no cover — needs fsaverage annot
        """The parcellation's cortical labels (unknown/medial-wall dropped) — shared by the dSPM inverse
        (`build_inverse`, 728) and the fNIRS-priored inverse's parcel aggregation (4so)."""
        _, subjects_dir = Source._fsaverage_dir()
        return [lbl for lbl in mne.read_labels_from_annot("fsaverage", self.cfg.parcellation,
                                                          subjects_dir=subjects_dir, verbose=False)
                if "unknown" not in lbl.name]

    def source_positions(self) -> np.ndarray:
        """3D positions `[n_src, 3]` of the fixed-orientation source-space vertices, in the forward's source
        order (concatenated over hemispheres) — the frame a spatial prior (e.g. fNIRS 'where', 4so) lives in."""
        fwd, _ = self.build_forward()
        fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True, verbose=False)
        return np.vstack([s["rr"][s["vertno"]] for s in fwd["src"]])   # [n_src, 3]

    def build_inverse(self):  # pragma: no cover — needs fsaverage template data
        """Build (or load from cache) the dSPM inverse operator + cortical labels for the montage.

        Returns `(inverse_operator, labels)`. The fsaverage forward solution is computed once per
        (montage, sfreq, config) and the inverse cached to `processed_dir()/source_operators/<key>-inv.fif`."""
        labels = self.cortical_labels()
        cache_dir = Config.processed_dir() / "source_operators"
        cache_dir.mkdir(parents=True, exist_ok=True)
        inv_path = cache_dir / f"{self._cache_key()}-inv.fif"
        if inv_path.exists():
            return mne.minimum_norm.read_inverse_operator(str(inv_path), verbose=False), labels

        logger.info(f"building fsaverage forward+inverse for {len(self.ch_names)} ch @ {self.sfreq} Hz "
                    f"({self.cfg.spacing})")
        fwd, info = self.build_forward()
        inverse = mne.minimum_norm.make_inverse_operator(info, fwd, mne.make_ad_hoc_cov(info), verbose=False)
        mne.minimum_norm.write_inverse_operator(str(inv_path), inverse, verbose=False)
        return inverse, labels

    def to_parcels(self, epochs: np.ndarray) -> np.ndarray:  # pragma: no cover — MNE fwd/inverse
        """Project sensor epochs `[n, ch, t]` to a source-space parcel series `[n, n_labels, t]` via the template
        dSPM inverse + label-time-course extraction. Anatomically-named, montage-independent — the substrate for
        source-space EEG↔fNIRS fusion (bd 728)."""
        inverse, labels = self.build_inverse()
        info = self._montage_info()
        ep = mne.EpochsArray(np.asarray(epochs, dtype=np.float64), info, verbose=False)
        stcs = mne.minimum_norm.apply_inverse_epochs(ep, inverse, lambda2=1.0 / self.cfg.snr ** 2,
                                                     method=self.cfg.method, verbose=False)
        pc = mne.extract_label_time_course(stcs, labels, inverse["src"], mode=self.cfg.label_mode, verbose=False)
        return np.asarray(pc, dtype=np.float32)                         # [n, n_labels, t]
