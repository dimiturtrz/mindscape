"""Paired EEG+fNIRS joint forward generator (bd shd) — the source-space fusion engine's ground truth.

One **shared** per-parcel neural latent `source[n, P, t]` drives BOTH modalities through modality-specific
forwards, so a fusion decoder can be scored against a known cortical source:
  - EEG  = a lead field (fsaverage template forward, bd 728) projecting the SIGNED parcel dipole moment to
           sensors  → `eeg[n, ch, t]`.
  - fNIRS = the POSITIVE neural envelope of the same parcels convolved with a double-gamma HRF (bd 7jn's
           independent forward), spread to optical channels by a geometric sensitivity `A[C, P]`, plus
           common-mode systemic physiology → `(hbo, hbr)[n, C, t]`.

This is the paired analog of the fNIRS-only `Synthetic` generator: there the drive was per-trial; here the
drive lives on cortical parcels shared with the EEG lead field, so EEG and fNIRS observe ONE latent. The
load-bearing approximation is `sensitivity` — a coarse geometric Gaussian "banana" (scalp optode → inward
cortical projection → distance-to-parcel Gaussian), NOT a Monte-Carlo photon model; it is enough for a
testbed that only needs "channels near an active parcel see it, distant ones don't" (research deep-dive
2026-07-15_joint_eeg_fnirs_forward_generator.md). The pure forwards are unit-tested; `generate` (which needs
the fsaverage template data) is the thin I/O shell.
"""
from __future__ import annotations

import mne
import numpy as np
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from scipy.signal import fftconvolve

from core.data.fnirs.synthetic import SynthConfig, Synthetic
from core.features.eeg.source import Source, SourceConfig


class Grid(BaseModel):
    """The shape + sample rate of one generate call: `n_trials` × `n_parcels` × `n_times` at `sfreq`. Bundled
    so the forwards take a shape object, not a positional (n, p, t, fs) clump. `n_parcels` defaults to the DK 68;
    `generate` overrides it from the actual lead field."""
    n_trials: int
    n_parcels: int = 68
    n_times: int = 128
    sfreq: float = 10.0


class JointConfig(BaseModel):
    """Joint-forward knobs. EEG side = dipole amplitude + sensor noise; fNIRS side = the geometric sensitivity
    width + the reused `SynthConfig` (HRF/systemic/noise); latent = how many parcels are active + burst shape."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    n_active: int = 3               # active parcels per trial — the shared ground truth support
    burst_hz: float = 10.0          # oscillatory burst frequency of the planted neural drive (alpha-ish)
    source_amp: float = 1.0         # planted parcel amplitude (arbitrary units; lead field/A are linear)
    eeg_noise: float = 0.1          # additive sensor-EEG noise std, relative to the projected signal std
    cortex_depth_m: float = 0.018   # inward scalp→cortex projection distance for an fNIRS channel's peak sensitivity
    sens_sigma_m: float = 0.025     # Gaussian sensitivity width (m): parcel-centroid ↔ channel cortical point
    synth: SynthConfig = SynthConfig()    # fNIRS HRF + systemic + measurement noise (reused, not duplicated)
    localize: SourceConfig = SourceConfig()   # EEG source-localization template knobs (spacing/method/parcellation)


class SharedLatent(BaseModel):
    """The planted ground truth: the per-parcel neural drive `source[n, P, t]` and, per trial, which parcel
    indices are active (`active[n, k]`). BOTH modalities are generated from `source`, so it IS the target a
    source-space fusion decoder must recover."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    source: Float[np.ndarray, "n p t"]
    active: Int[np.ndarray, "n k"]


class JointForward:
    """Joint EEG+fNIRS forward operator (bd shd). Pure forwards (`plant_latent`, `sensitivity`,
    `eeg_from_source`, `fnirs_from_source`) take explicit lead field / geometry so they unit-test without the
    fsaverage template; `generate` composes them onto the real template forward (bd 728)."""

    @staticmethod
    def plant_latent(grid: Grid, cfg: JointConfig | None = None, seed: int = 0) -> SharedLatent:
        """Plant `cfg.n_active` random active parcels per trial, each carrying a windowed `burst_hz` oscillation.
        Returns the shared latent `source[n, P, t]` (zero on silent parcels) + the active-parcel indices."""
        cfg = cfg or JointConfig()
        rng = np.random.default_rng(seed)
        t = np.arange(grid.n_times) / grid.sfreq
        # smooth on/offset — a burst, not a full-length tone
        window = np.hanning(grid.n_times)
        source = np.zeros((grid.n_trials, grid.n_parcels, grid.n_times), dtype=np.float32)
        active = np.empty((grid.n_trials, cfg.n_active), dtype=np.int64)
        for i in range(grid.n_trials):
            picks = rng.choice(grid.n_parcels, size=cfg.n_active, replace=False)
            active[i] = picks
            phase = rng.uniform(0, 2 * np.pi, size=cfg.n_active)
            burst = cfg.source_amp * window[None, :] * np.sin(2 * np.pi * cfg.burst_hz * t[None, :] + phase[:, None])
            source[i, picks] = burst.astype(np.float32)
        return SharedLatent(source=source, active=active)

    @staticmethod
    def sensitivity(parcel_xyz: Float[np.ndarray, "p 3"], channel_xyz: Float[np.ndarray, "c 3"],
                    cfg: JointConfig | None = None) -> Float[np.ndarray, "c p"]:
        """Coarse geometric optical sensitivity `A[C, P]`: project each scalp channel inward toward the head
        centroid by `cortex_depth_m` to a cortical point, then weight parcel→channel by a Gaussian of the
        centroid↔point distance (`sens_sigma_m`). Rows are the "banana" a channel integrates over. A defensible
        approximation to the photon-transport Jacobian for a validation testbed (not Monte-Carlo)."""
        cfg = cfg or JointConfig()
        center = parcel_xyz.mean(axis=0)                               # head centroid ≈ mean cortical position
        inward = center[None, :] - channel_xyz
        inward /= (np.linalg.norm(inward, axis=1, keepdims=True) + 1e-12)
        cortical_pt = channel_xyz + cfg.cortex_depth_m * inward        # [C, 3] each channel's peak-sensitivity point
        d = np.linalg.norm(cortical_pt[:, None, :] - parcel_xyz[None, :, :], axis=2)   # [C, P]
        return np.exp(-0.5 * (d / cfg.sens_sigma_m) ** 2).astype(np.float32)

    @staticmethod
    def eeg_from_source(source: Float[np.ndarray, "n p t"], lead_field: Float[np.ndarray, "ch p"],
                        cfg: JointConfig | None = None, seed: int = 0) -> Float[np.ndarray, "n ch t"]:
        """Forward-project the SIGNED parcel dipole moment to sensor EEG via a parcel-reduced lead field, plus
        additive sensor noise scaled to the projected-signal std (`eeg_noise`)."""
        cfg = cfg or JointConfig()
        rng = np.random.default_rng(seed)
        eeg = np.einsum("cp,npt->nct", lead_field, source)             # lead-field mixing to sensors
        noise = rng.standard_normal(eeg.shape) * (cfg.eeg_noise * (eeg.std() + 1e-12))
        return (eeg + noise).astype(np.float32)

    @staticmethod
    def fnirs_from_source(source: Float[np.ndarray, "n p t"], sensitivity: Float[np.ndarray, "c p"],
                          fs: float, cfg: JointConfig | None = None, seed: int = 0
                          ) -> tuple[Float[np.ndarray, "n c t"], Float[np.ndarray, "n c t"]]:
        """Forward-generate paired `(hbo, hbr)[n, C, t]` from the SAME shared latent: the POSITIVE neural
        envelope `|source|` per parcel ⊛ double-gamma HRF = a clean per-parcel hemodynamic response, spread to
        channels by `sensitivity`, then per-channel common-mode systemic + measurement noise (reusing the
        independent `Synthetic` forward, bd 7jn). HbR is the anti-correlated `-hbr_ratio·HbO` neural part."""
        cfg = cfg or JointConfig()
        scfg = cfg.synth
        rng = np.random.default_rng(seed)
        n, _, length = source.shape
        hrf = Synthetic.double_gamma_hrf(fs, scfg)
        drive = np.abs(source)                                         # neural→blood couples to activity magnitude
        resp = fftconvolve(drive, hrf[None, None, :], axes=2)[..., :length]     # [n, P, t] clean response
        chan = np.einsum("cp,npt->nct", sensitivity, resp)             # spread parcels → optical channels
        n_ch = sensitivity.shape[0]
        sys_o = Synthetic._systemic(n * n_ch, length, fs, scfg, rng).reshape(n, n_ch, length)
        sys_r = sys_o + Synthetic._systemic(n * n_ch, length, fs, scfg, rng).reshape(n, n_ch, length) * 0.15
        hbo = chan + sys_o + rng.standard_normal(chan.shape) * scfg.noise_std
        hbr = -scfg.hbr_ratio * chan + sys_r + rng.standard_normal(chan.shape) * scfg.noise_std
        return hbo.astype(np.float32), hbr.astype(np.float32)

    @staticmethod
    def _parcel_lead_field(fwd, labels, src_pos: Float[np.ndarray, "s 3"]
                           ) -> tuple[Float[np.ndarray, "ch p"], Float[np.ndarray, "p 3"]]:
        """Reduce the fixed-orientation per-vertex lead field to per-parcel by averaging each label's source
        columns; parcel position = its vertices' mean. Returns `(lead[ch, P], parcel_xyz[P, 3])`."""
        fixed = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True, verbose=False)
        gain = fixed["sol"]["data"]                                    # [n_ch, n_src]
        vertno = [s["vertno"] for s in fixed["src"]]
        offsets = np.cumsum([0, *[len(v) for v in vertno]])
        lead_cols, parcel_pos = [], []
        for lbl in labels:
            hemi = 0 if lbl.hemi == "lh" else 1
            idx = np.searchsorted(vertno[hemi], np.intersect1d(lbl.vertices, vertno[hemi]))
            if len(idx) == 0:
                continue
            cols = offsets[hemi] + idx
            lead_cols.append(gain[:, cols].mean(axis=1))
            parcel_pos.append(src_pos[cols].mean(axis=0))
        return np.stack(lead_cols, axis=1).astype(np.float32), np.stack(parcel_pos).astype(np.float32)

    @staticmethod
    def generate(montage: tuple[list[str], float], fnirs_xyz: Float[np.ndarray, "c 3"], grid: Grid,
                 cfg: JointConfig | None = None, seed: int = 0) -> dict:  # pragma: no cover — fsaverage template
        """Compose the pure forwards onto the real fsaverage template lead field (bd 728). `montage` = the EEG
        `(ch_names, sfreq)`; `grid` = the trial/time shape (its `n_parcels` is overridden by the actual DK count).
        Returns `{eeg[n,ch,t], hbo, hbr [n,C,t], source[n,P,t], active}` — paired observations + shared ground
        truth. `fnirs_xyz` = optical-channel scalp positions (m, head frame); EEG + fNIRS share the P DK parcels.
        First cut = ONE shared `grid.sfreq` grid for both modalities (the HRF is a heavy low-pass, so a coarse-vs-
        fine EEG/fNIRS rate split is a follow-up decimation, not a correctness issue for the shared latent)."""
        cfg = cfg or JointConfig()
        ch_names, sfreq = montage
        src = Source(ch_names, sfreq, cfg.localize)
        fwd, _ = src.build_forward()
        labels = src.cortical_labels()
        lead, parcel_xyz = JointForward._parcel_lead_field(fwd, labels, src.source_positions())
        sens = JointForward.sensitivity(parcel_xyz, np.asarray(fnirs_xyz, dtype=np.float32), cfg)
        latent = JointForward.plant_latent(grid.model_copy(update={"n_parcels": lead.shape[1]}), cfg, seed)
        eeg = JointForward.eeg_from_source(latent.source, lead, cfg, seed)
        hbo, hbr = JointForward.fnirs_from_source(latent.source, sens, grid.sfreq, cfg, seed)
        return {"eeg": eeg, "hbo": hbo, "hbr": hbr, "source": latent.source, "active": latent.active}
