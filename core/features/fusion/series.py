"""The pre-raster cross-modal representation — paired EEG/fNIRS on one lag-aligned time grid.

EEG → θ/α/β band-power envelopes (the fast electrical strength), fNIRS → CBSI neural (the slow blood response),
with the hemodynamic offset DERIVED per block (`estimate_coupling`). This is the shared series that the viz
rasterizes (camera) and the coupling diagnostics consume — one home for the envelope + lag-align logic.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Float
from pydantic import BaseModel
from scipy.interpolate import interp1d
from scipy.signal import butter, hilbert, sosfiltfilt

from core.features.fnirs.chromophore import Chromophore
from core.features.fusion.coupling import Coupling

_THETA, _ALPHA, _BETA = (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)
_BANDS = {"theta": _THETA, "alpha": _ALPHA, "beta": _BETA}


class SeriesConfig(BaseModel):
    """Timing knobs for the lag-aligned cross-modal series. `fs_e`/`fs_f` = EEG/fNIRS sample rates; `tmin_f` =
    the fNIRS epoch start (s); `fps` = the shared display grid rate; `t_end` = display window length (s);
    `lag_s` = a fixed hemodynamic offset override (None = derive it via `estimate_coupling`)."""
    fs_e: float = 100.0
    fs_f: float = 10.0
    tmin_f: float = -2.0
    fps: float = 10.0
    t_end: float = 20.0
    lag_s: float | None = None


class Series:
    """The pre-raster cross-modal representation — paired EEG/fNIRS on one lag-aligned time grid (free helpers
    folded in as staticmethods, public names kept)."""

    @classmethod
    def band_env(cls, X: Float[np.ndarray, "n ch t"], fs: float,
                 band: tuple[float, float]) -> Float[np.ndarray, "n ch t"]:
        """Band-power envelope per channel: bandpass → analytic-signal magnitude → `[n, ch, t]`. The slow envelope
        is what carries cognitive-state info, so it (not raw EEG) is the fast-layer feature."""
        sos = butter(4, [band[0], band[1]], btype="band", fs=fs, output="sos")
        return np.abs(hilbert(sosfiltfilt(sos, X, axis=-1), axis=-1))

    @classmethod
    def resample_time(cls, X: Float[np.ndarray, "n ch t"], t_src: Float[np.ndarray, "t"],
                      t_dst: Float[np.ndarray, "t2"]) -> Float[np.ndarray, "n ch t2"]:
        """Linear-resample `X[n, ch, t]` from source time axis `t_src` to target `t_dst` (vectorized over axis -1)."""
        return interp1d(t_src, X, axis=-1, bounds_error=False, fill_value=0.0)(t_dst).astype(X.dtype)

    @classmethod
    def channel_series(cls, Xe, Xf, config: SeriesConfig | None = None):
        """The pre-raster fused representation — the single source of truth both the decoder tensor and the viz
        consume. Paired EEG/fNIRS -> per-channel activity on ONE lag-aligned time grid:
          eeg    = {theta,alpha,beta} band-power envelopes `[n, ch_e, T]`  (fast electrical STRENGTH),
          neural = CBSI(HbO,HbR) `[n, ch_f, T]`  (slow blood response, systemic-rejected — origin + spread).
        The hemodynamic offset is **derived** (`estimate_coupling`, EEG-β envelope vs CBSI) unless `config.lag_s` is
        given as a fixed override — no magic 5 s. Blood LAGS neural (`blood(t) ≈ neural(t−lag)`), so to recover the
        neural activity at display-time τ we read the blood FORWARD, at `τ+lag` (sample fNIRS at `tf−lag`) — this
        aligns the blood to the EEG event that drove it and fills the EARLY window (τ=0 pulls the blood at +lag,
        which exists); the unrecorded tail (τ > t_end−lag needs blood past the recording) zero-fills instead.
        Returns `(eeg, neural, t_dst, coupling)`, coupling = {lag, decay, beta}."""
        cfg = config or SeriesConfig()
        ch_f = Xf.shape[1] // 2
        t_dst = np.arange(0, cfg.t_end, 1.0 / cfg.fps)
        te = np.arange(Xe.shape[2]) / cfg.fs_e
        eeg = {name: cls.resample_time(cls.band_env(Xe, cfg.fs_e, band), te, t_dst)
               for name, band in _BANDS.items()}
        tf = cfg.tmin_f + np.arange(Xf.shape[2]) / cfg.fs_f
        if cfg.lag_s is None:
            neural0 = Chromophore.cbsi_neural(cls.resample_time(Xf[:, :ch_f, :], tf, t_dst),   # zero-lag CBSI
                                              cls.resample_time(Xf[:, ch_f:, :], tf, t_dst))
            lag, decay, beta = Coupling.estimate_coupling(eeg["beta"].mean(1), neural0.mean(1), cfg.fps)  # β ~ +blood
        else:
            lag, decay, beta = float(cfg.lag_s), float("nan"), float("nan")
        hbo = cls.resample_time(Xf[:, :ch_f, :], tf - lag, t_dst)  # read blood forward (τ+lag) to align to neural
        hbr = cls.resample_time(Xf[:, ch_f:, :], tf - lag, t_dst)
        return eeg, Chromophore.cbsi_neural(hbo, hbr), t_dst, {"lag": lag, "decay": decay, "beta": beta}
