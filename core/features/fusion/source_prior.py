"""fNIRS-informed EEG source imaging (bd 4so) — the physically-correct complementarity: fNIRS says WHERE the
cortex is active, EEG resolves WHEN. Regularize the ill-posed EEG inverse with an fNIRS activation prior so the
minimum-norm solution is drawn toward fNIRS-active patches, in a shared cortical frame.

This is the classic fMRI-informed inverse (Liu, Belliveau & Dale 1998) with fNIRS as the spatial prior: a
per-source prior variance `R = diag(w)` (large where fNIRS is active, a nonzero floor elsewhere so EEG can still
place sources fMRI/fNIRS missed) gives the weighted minimum-norm operator

    K = R Gᵀ (G R Gᵀ + λ² I)⁻¹ ,   λ² = tr(G R Gᵀ) / n_ch / snr²

with `G` the lead field ([`core.features.eeg.source.build_forward`], 728). The operator is pure linear algebra
(`weighted_min_norm_inverse`, unit-tested); `source_estimate_with_prior` wraps it on the fsaverage lead field.
The fNIRS-activation → per-source weight mapping (optode co-registration) is the data side, consumed here as `w`.
"""
from __future__ import annotations

import mne
import numpy as np

from core.features.eeg.source import SourceConfig, build_forward, cortical_labels


def weighted_min_norm_inverse(leadfield: np.ndarray, source_prior: np.ndarray, snr: float = 3.0) -> np.ndarray:
    """fMRI/fNIRS-informed weighted minimum-norm inverse `K = R Gᵀ (G R Gᵀ + λ² I)⁻¹` (Liu 1998).

    `leadfield` `G [n_ch, n_src]` (fixed-orientation gain), `source_prior` `w [n_src]` ≥ 0 = the per-source prior
    variance (fNIRS activation, with a nonzero floor). Returns `K [n_src, n_ch]`: `source = K @ sensor`. Sources
    in high-prior regions get larger prior variance, so the ill-posed inverse attributes sensor activity there
    preferentially — the fNIRS 'where' regularizing the EEG solution."""
    g = np.asarray(leadfield, dtype=np.float64)
    w = np.asarray(source_prior, dtype=np.float64)
    if w.shape[0] != g.shape[1]:
        raise ValueError(f"source_prior length {w.shape[0]} != n_src {g.shape[1]}")
    if np.any(w < 0):
        raise ValueError("source_prior must be non-negative (it is a prior variance)")
    r_gt = w[:, None] * g.T                                     # diag(w) @ Gᵀ  -> [n_src, n_ch]
    grgt = g @ r_gt                                             # G R Gᵀ         -> [n_ch, n_ch]
    lam2 = np.trace(grgt) / grgt.shape[0] / snr ** 2            # trace-normalized regularization
    return r_gt @ np.linalg.inv(grgt + lam2 * np.eye(grgt.shape[0]))


def _parcel_aggregator(src, labels) -> np.ndarray:
    """`A [n_labels, n_src]` averaging each label's sources — source order = concat of `src[h]["vertno"]`.
    Maps the fixed-orientation source estimate onto the same Desikan-Killiany parcels the dSPM path uses
    (`source.to_parcels`), so fNIRS-priored and plain-dSPM decodes are compared on one representation."""
    idx, start = [], 0
    for s in src:
        idx.append({int(v): start + i for i, v in enumerate(s["vertno"])})
        start += len(s["vertno"])
    a = np.zeros((len(labels), start))
    for li, lbl in enumerate(labels):
        m = idx[0 if lbl.hemi == "lh" else 1]
        ids = [m[int(v)] for v in lbl.vertices if int(v) in m]
        if ids:
            a[li, ids] = 1.0 / len(ids)
    return a


def prior_leadfield(ch_names: list[str], sfreq: float,
                    cfg=None) -> tuple[np.ndarray, np.ndarray]:   # pragma: no cover — MNE fsaverage lead field
    """`(g [n_ch, n_src], aggregator [n_labels, n_src])` for a montage — the fixed-orientation lead field and
    the Desikan-Killiany parcel-averaging matrix. The expensive forward is built once here so a per-subject
    decode can vary only the prior `w` (`weighted_min_norm_inverse`) without rebuilding it (4so batch decode)."""
    cfg = cfg or SourceConfig()
    fwd, _ = build_forward(ch_names, sfreq, cfg)
    fwd = mne.convert_forward_solution(fwd, force_fixed=True, use_cps=True, verbose=False)
    return fwd["sol"]["data"], _parcel_aggregator(fwd["src"], cortical_labels(cfg))


def parcels_from_leadfield(epochs: np.ndarray, leadfield: np.ndarray, aggregator: np.ndarray,
                           source_prior: np.ndarray | None = None, snr: float = 3.0) -> np.ndarray:
    """Priored-inverse parcel series `[n, n_labels, t]` from a precomputed `leadfield`/`aggregator`
    (`prior_leadfield`) — the inner op the 4so decode loops per subject, varying only `source_prior`."""
    w = np.ones(leadfield.shape[1]) if source_prior is None else np.asarray(source_prior, dtype=np.float64)
    k = weighted_min_norm_inverse(leadfield, w, snr)
    m = aggregator @ k                                         # [n_labels, n_ch] — fold inverse+aggregation
    x = np.asarray(epochs, dtype=np.float64)
    return np.einsum("lc,nct->nlt", m, x).astype(np.float32)   # [n, n_labels, t]
