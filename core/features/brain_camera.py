"""Multi-band 'brain-camera' tensor — restructure paired EEG+fNIRS into a shared 2D-surface × time video.

The honest fusion (not the flat concat that diluted): both modalities only see the outer cortical shell, so
model the *surface* — a 2D head grid — with time. Each block becomes `X[C, H, W, T]`:
  - EEG  → θ/α/β **band-power envelope** maps (the fast electrical layers),
  - fNIRS → HbO/HbR maps (the slow metabolic layers), **lag-shifted** to align with the EEG event that drove
    them (the hemodynamic response trails ~5 s),
each interpolated from its sparse channels onto one 10-20-referenced grid, with a per-modality **validity
mask** (EEG and fNIRS cover the head differently — don't pretend a modality exists where it has no sensor).

Both montages live in the same 10-20 frame, so co-registration is free (normalize each to the unit head disk).
The overlap window is [0, tmax_common] s (EEG epoch 0-40 s, fNIRS -2→20 s → shared 0-20 s). Grid + fps kept
modest so the tensor fits in RAM (`16×16 × 10 Hz × 20 s` ≈ 2 MB/block f32).
"""
from __future__ import annotations

import numpy as np

_THETA, _ALPHA, _BETA = (4.0, 8.0), (8.0, 13.0), (13.0, 30.0)
_HRF_WIDTH = (2.0, 5.0)      # physiological hemodynamic dispersion (s): HRF FWHM ~5 s -> std ~2-5 s. A width
                             # FLOOR stops the coupling fit railing to a degenerate spike on weak/short data.


def eeg_positions(ch_names: list[str]) -> np.ndarray:
    """2D scalp positions for EEG channels via the MNE standard 10-05 montage, normalized to the unit disk.
    Returns `[n_ch, 2]` in the same head-disk convention as `fnirs_positions`."""
    import mne
    m = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    pos = np.array([m[c][:2] if c in m else (np.nan, np.nan) for c in ch_names], dtype=float)  # x,y (drop z)
    return _to_unit_disk(pos)


def fnirs_positions(subject_dir) -> np.ndarray:
    """2D positions of the 36 fNIRS channels from `mnt_nback.mat` (already a head-normalized 2D projection),
    normalized to the unit disk to match the EEG frame."""
    import scipy.io as sio
    mnt = sio.loadmat(subject_dir / "mnt_nback.mat", struct_as_record=False, squeeze_me=True)["mnt_nback"]
    pos = np.stack([np.asarray(mnt.x, dtype=float), np.asarray(mnt.y, dtype=float)], axis=1)  # [36, 2]
    return _to_unit_disk(pos)


def _to_unit_disk(pos: np.ndarray) -> np.ndarray:
    """Center + scale a 2D montage so its channels fit the unit disk (radius ≤ 1) — a common head frame."""
    p = pos - np.nanmean(pos, axis=0)
    r = np.nanmax(np.hypot(p[:, 0], p[:, 1]))
    return p / (r + 1e-9)


def _band_env(X: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Band-power envelope per channel: bandpass → analytic-signal magnitude → `[n, ch, t]`. The slow envelope
    is what carries cognitive-state info, so it (not raw EEG) is the fast-layer feature."""
    from scipy.signal import butter, hilbert, sosfiltfilt
    sos = butter(4, [band[0], band[1]], btype="band", fs=fs, output="sos")
    return np.abs(hilbert(sosfiltfilt(sos, X, axis=-1), axis=-1))


def _resample_time(X: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Linear-resample `X[n, ch, t]` from source time axis `t_src` to target `t_dst` (vectorized over axis -1)."""
    from scipy.interpolate import interp1d
    return interp1d(t_src, X, axis=-1, bounds_error=False, fill_value=0.0)(t_dst).astype(X.dtype)


def _bary(pos: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric linear-interpolation weight matrix `W[len(pts), n_ch]` mapping channel values at `pos` to
    arbitrary query points `pts` (Delaunay triangulation, computed once — every frame is then a matmul), plus
    the distance `d[len(pts)]` from each query point to its nearest sensor. Points outside the hull get zero
    weight. Channels with non-finite positions are dropped from the triangulation (weight column stays zero)."""
    from scipy.spatial import Delaunay
    ok = np.isfinite(pos).all(1)
    idx = np.where(ok)[0]
    p = pos[ok]
    tri = Delaunay(p)
    simp = tri.find_simplex(pts)                                        # [-1] outside hull
    W = np.zeros((len(pts), len(pos)), dtype=np.float32)
    inside = simp >= 0
    T = tri.transform[simp[inside]]
    b = np.einsum("nij,nj->ni", T[:, :2, :], pts[inside] - T[:, 2, :])  # first 2 barycentric coords
    bary = np.concatenate([b, 1 - b.sum(1, keepdims=True)], axis=1)     # [Nin, 3]
    verts = idx[tri.simplices[simp[inside]]]                            # original channel indices
    rows = np.where(inside)[0]
    for j in range(3):
        W[rows, verts[:, j]] = bary[:, j]
    d = np.sqrt(((pts[:, None, :] - p[None, :, :]) ** 2).sum(-1)).min(1)
    return W, d


def _interp_weights(pos: np.ndarray, grid: int) -> tuple[np.ndarray, np.ndarray]:
    """Barycentric weights `W[grid², n_ch]` onto the head grid + a validity `mask[grid²]` (pixel within ~1.5
    grid-spacings of a sensor)."""
    gx = np.linspace(-1, 1, grid)
    gxx, gyy = np.meshgrid(gx, gx)
    pts = np.stack([gxx.ravel(), gyy.ravel()], axis=1)                  # [G², 2]
    W, d = _bary(pos, pts)
    mask = (d <= (2.0 / grid) * 1.5).astype(np.float32)                 # [G²]
    return W, mask


def _apply(vals: np.ndarray, W: np.ndarray, grid: int) -> np.ndarray:
    """`vals[n, ch, t]` × `W[grid², ch]` -> `maps[n, grid, grid, t]` (one einsum, all frames at once)."""
    m = np.einsum("nct,gc->ngt", vals, W)                              # [n, G², t]
    n, _, t = m.shape
    return m.reshape(n, grid, grid, t)


_BANDS = {"theta": _THETA, "alpha": _ALPHA, "beta": _BETA}


def cbsi_neural(hbo: np.ndarray, hbr: np.ndarray) -> np.ndarray:
    """CBSI neural map (Cui 2010) — activation makes HbO/HbR anti-correlated, motion/systemic makes them
    common-mode; `HbO − α·HbR` (α = std(HbO)/std(HbR)) keeps the neural part, cancels the systemic. Uses BOTH
    chromophores — the whole point of two wavelengths. `hbo`/`hbr` are `[n, ch, t]` -> `[n, ch, t]`."""
    a = hbo.std(axis=2, keepdims=True) / (hbr.std(axis=2, keepdims=True) + 1e-9)
    return 0.5 * (hbo - a * hbr)


def coverage_map(pos_e: np.ndarray, pos_f: np.ndarray, grid: int) -> np.ndarray:
    """Locality-coverage confidence `[grid, grid]` — EEG↔fNIRS coupling is LOCAL, so a pixel's joint signal is
    only trustworthy where a co-located pair exists (near BOTH an EEG and an fNIRS sensor). Gaussian falloff to
    the nearest sensor of each modality; product dims where either modality has no nearby sensor."""
    gx = np.linspace(-1, 1, grid)
    gxx, gyy = np.meshgrid(gx, gx)
    pts = np.stack([gxx.ravel(), gyy.ravel()], axis=1)                  # [G², 2]
    s2 = (0.20 ** 2) * 2                                                # co-sampling radius² (unit-disk units)

    def near(pos):
        p = pos[np.isfinite(pos).all(1)]
        return np.exp(-((pts[:, None, :] - p[None, :, :]) ** 2).sum(-1).min(1) / s2)
    return (near(pos_e) * near(pos_f)).reshape(grid, grid).astype(np.float32)


def _gamma_kernel(t: np.ndarray, peak: float, width: float) -> np.ndarray:
    """Causal single-gamma hemodynamic kernel on time axis `t` (s), parameterized by its center-of-mass `peak`
    (the delay) and dispersion `width` (both seconds), normalized to unit area. `mean = a·b = peak`,
    `std = √a·b = width` -> shape `a = (peak/width)²`, scale `b = width²/peak`."""
    a = (peak / width) ** 2
    b = width ** 2 / peak
    g = np.where(t > 0, np.power(np.clip(t, 1e-6, None), a - 1.0) * np.exp(-t / b), 0.0)
    s = g.sum()
    return g / s if s > 0 else g


def estimate_coupling(drive: np.ndarray, resp: np.ndarray, fs: float, *, lag_max: float = 12.0,
                      klen: float = 30.0):
    """Derive the EEG→blood coupling from the data instead of hardcoding a 5 s shift. `drive` = EEG band-power
    envelope, `resp` = fNIRS CBSI, both `[n, T]` global (channel-mean) on ONE **zero-lag** grid at rate `fs`.
    Fit a causal gamma kernel `g(peak, width)` maximizing the correlation between the EEG-**predicted** blood
    `drive ⊛ g` and the measured `resp` (grid search: delay 2-12 s; dispersion constrained to the physiological
    HRF range `_HRF_WIDTH` — an HRF is a smooth bump, NOT a spike, so a width floor keeps the fit from railing to
    a degenerate delta on weak/short data: physics > statistics). Returns `(lag, decay, beta)`: `lag` = kernel
    center-of-mass (s, the offset), `decay` = tail time-constant `b` (s, the smearing), `beta` = least-squares
    gain (EEG-envelope → CBSI unit bridge). Self-calibrates per subject — neurovascular latency varies."""
    from scipy.signal import fftconvolve
    d = drive.astype(np.float64)
    r = resp.astype(np.float64)
    rz = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + 1e-9)
    tk = np.arange(0, klen, 1.0 / fs)
    best = (-1.0, 6.0, _HRF_WIDTH[0])                                  # (|corr|², peak, width)
    for peak in np.arange(2.0, lag_max + 1e-9, 0.5):
        for width in np.arange(_HRF_WIDTH[0], min(_HRF_WIDTH[1], peak) + 1e-9, 0.5):
            g = _gamma_kernel(tk, peak, width)
            pred = fftconvolve(d, g[None, :], axes=1)[:, :d.shape[1]]
            pz = (pred - pred.mean(1, keepdims=True)) / (pred.std(1, keepdims=True) + 1e-9)
            score = float((pz * rz).mean(1).mean()) ** 2               # coupling STRENGTH (sign-agnostic — β
            if score > best[0]:                                       # carries the direction; don't assume +)
                best = (score, peak, width)
    _, peak, width = best
    pred = fftconvolve(d, _gamma_kernel(tk, peak, width)[None, :], axes=1)[:, :d.shape[1]]
    beta = float((pred * r).sum() / ((pred ** 2).sum() + 1e-12))       # raw LS gain (units bridge)
    return float(peak), float(width ** 2 / peak), beta


def channel_series(Xe, Xf, *, fs_e=100.0, fs_f=10.0, tmin_f=-2.0, fps=10.0, t_end=20.0, lag_s=None):
    """The pre-raster fused representation — the single source of truth both the decoder tensor and the viz
    consume. Paired EEG/fNIRS -> per-channel activity on ONE lag-aligned time grid:
      eeg    = {theta,alpha,beta} band-power envelopes `[n, ch_e, T]`  (fast electrical STRENGTH),
      neural = CBSI(HbO,HbR) `[n, ch_f, T]`  (slow blood response, systemic-rejected — origin + spread).
    The hemodynamic offset is **derived** (`estimate_coupling`, EEG-β envelope vs CBSI) unless `lag_s` is given
    as a fixed override — no magic 5 s. Blood LAGS neural (`blood(t) ≈ neural(t−lag)`), so to recover the neural
    activity at display-time τ we read the blood FORWARD, at `τ+lag` (sample fNIRS at `tf−lag`) — this aligns the
    blood to the EEG event that drove it and fills the EARLY window (τ=0 pulls the blood at +lag, which exists);
    the unrecorded tail (τ > t_end−lag needs blood past the recording) zero-fills instead. Returns
    `(eeg, neural, t_dst, coupling)`, coupling = {lag, decay, beta}."""
    ch_f = Xf.shape[1] // 2
    t_dst = np.arange(0, t_end, 1.0 / fps)
    te = np.arange(Xe.shape[2]) / fs_e
    eeg = {name: _resample_time(_band_env(Xe, fs_e, band), te, t_dst) for name, band in _BANDS.items()}
    tf = tmin_f + np.arange(Xf.shape[2]) / fs_f
    if lag_s is None:
        neural0 = cbsi_neural(_resample_time(Xf[:, :ch_f, :], tf, t_dst),      # zero-lag CBSI for the estimate
                              _resample_time(Xf[:, ch_f:, :], tf, t_dst))
        lag, decay, beta = estimate_coupling(eeg["beta"].mean(1), neural0.mean(1), fps)  # β power ~ +blood
    else:
        lag, decay, beta = float(lag_s), float("nan"), float("nan")
    hbo = _resample_time(Xf[:, :ch_f, :], tf - lag, t_dst)         # read blood forward (τ+lag) to align to neural
    hbr = _resample_time(Xf[:, ch_f:, :], tf - lag, t_dst)
    return eeg, cbsi_neural(hbo, hbr), t_dst, {"lag": lag, "decay": decay, "beta": beta}


def build_tensor(Xe, Xf, pos_e, pos_f, *, grid=16, **kw):
    """Paired EEG+fNIRS -> the brain-camera decoder tensor `X[n, C, grid, grid, T]`, C = [θ, α, β, CBSI-neural,
    coverage]. The PRINCIPLED channels (CBSI neural — both chromophores, systemic-rejected; locality coverage),
    rasterized onto one shared 10-20 grid. The model learns the fusion from these components — we don't hardcode
    the joint product (that's the viz's display choice). `kw` -> channel_series (fs/fps/lag/window)."""
    eeg, neural, _, _ = channel_series(Xe, Xf, **kw)
    W_e, _ = _interp_weights(pos_e, grid)
    W_f, _ = _interp_weights(pos_f, grid)
    eeg_maps = [_zscore(_apply(eeg[b], W_e, grid)) for b in _BANDS]                 # strength layers
    neural_map = _zscore(_apply(neural, W_f, grid))                                 # origin + spread
    cov = coverage_map(pos_e, pos_f, grid)                                          # locality gate
    stack = eeg_maps + [neural_map, _broadcast(cov, eeg_maps[0].shape)]
    return np.stack(stack, axis=1).astype(np.float32)                              # [n, C=5, grid, grid, T]


def fused_node_series(Xe, Xf, pos_e, pos_f, *, band="sum", **kw):
    """The FUSION-only signal, collapsed to EEG channel format `[n, n_e, T]` — so the strong EEG decoder
    (per-subject re-centered tangent-space) can eat it directly. At each EEG sensor node the joint = EEG
    electrical STRENGTH × the co-located fNIRS neural (CBSI, interpolated to that node) × locality COVERAGE
    (Gaussian to the nearest fNIRS sensor). This is the genuinely-cross-modal quantity — not raw EEG, not raw
    fNIRS — so its spatial covariance carries the coupled firing pattern, nothing a single modality already has.
    `band`='sum' (total electrical strength) or a single band name. `kw` -> channel_series. Returns `(joint,
    coupling)`; nodes with non-finite EEG positions are dropped (kept consistent by the fixed montage)."""
    eeg, neural, _, coupling = channel_series(Xe, Xf, **kw)
    ok = np.isfinite(pos_e).all(1)
    strength = sum(eeg[b] for b in _BANDS) if band == "sum" else eeg[band]      # [n, n_e, T], envelopes ≥ 0
    strength = strength[:, ok, :]
    Wq, _ = _bary(pos_f, pos_e[ok])                                             # fNIRS -> EEG-node interp [n_e, n_f]
    neural_e = np.einsum("nct,gc->ngt", neural, Wq)                             # CBSI at the EEG nodes [n, n_e, T]
    s2 = (0.20 ** 2) * 2
    pf = pos_f[np.isfinite(pos_f).all(1)]
    d2 = ((pos_e[ok][:, None, :] - pf[None, :, :]) ** 2).sum(-1).min(1)         # dist² each EEG node -> nearest fNIRS
    cov_e = np.exp(-d2 / s2)                                                    # locality coverage per node [n_e]
    joint = strength * neural_e * cov_e[None, :, None]                         # [n, n_e, T]
    return joint.astype(np.float32), coupling


def _zscore(m: np.ndarray) -> np.ndarray:
    """Z-score a map block over space+time per sample (comparability across bands/modalities)."""
    mu = m.mean(axis=(1, 2, 3), keepdims=True)
    sd = m.std(axis=(1, 2, 3), keepdims=True) + 1e-6
    return (m - mu) / sd


def _broadcast(mask2d: np.ndarray, like_shape) -> np.ndarray:
    """A `[grid,grid]` mask -> `[n, grid, grid, T]` matching a map block."""
    n, g, _, t = like_shape
    return np.broadcast_to(mask2d[None, :, :, None], (n, g, g, t)).astype(np.float32)
