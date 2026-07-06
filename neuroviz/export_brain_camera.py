"""Export brain-camera data for neuroviz — per-channel EEG band-power + fNIRS HbO time-series for one example
block, on a shared time grid (fNIRS lag-aligned), with both montages' 2D positions. The web view interpolates
these to head-maps (EEG top-left, fNIRS bottom-left, fused right) and animates them.

    python -m neuroviz.export_brain_camera --subject 1 --block 0
Writes neuroviz/web/data/brain_camera.json (gitignored, like the other view data).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.config import REPO
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.fnirs.base import FnirsCfg
from core.data.fnirs import shin2017 as fnmod
from core.features import brain_camera as bc

_FS_E, _FS_F, _TMIN_F = 100.0, 10.0, -2.0
_FPS, _TEND, _FN_TMAX = 10.0, 20.0, 32.0        # fNIRS epoched past _TEND so read-forward (τ+lag) fills the tail
_COV_GRID = 40                                   # locality-coverage kernel resolution exported for the viz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()

    me = store.load("shin2017_nback_eeg", EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=_FS_E))
    mf = store.load("shin2017_nback", FnirsCfg(tmax=_FN_TMAX))     # past 20 s so the read-forward tail has blood
    s = str(args.subject)
    Xe, ye = store.gather(me.filter(me["subject"] == s))
    Xf, yf = store.gather(mf.filter(mf["subject"] == s))
    assert np.array_equal(ye, yf), "EEG/fNIRS misaligned"
    b = args.block
    pos_e = bc.eeg_positions(eegmod.adapter().channels())
    pos_f = bc.fnirs_positions(fnmod.adapter()._subject_dir(args.subject))

    # single source of truth: core computes the fused representation (band-power envelopes + CBSI neural,
    # lag-aligned) and the locality-coverage kernel. The viz just displays them — no fusion logic in JS.
    # Derive the hemodynamic coupling (offset + decay) over ALL the subject's blocks (robust), then export the
    # requested block aligned by that derived lag — no fixed 5 s.
    *_, coupling = bc.channel_series(Xe, Xf, fs_e=_FS_E, fs_f=_FS_F, tmin_f=_TMIN_F, fps=_FPS, t_end=_TEND)
    eeg_s, neural_s, t_dst, _ = bc.channel_series(Xe[b:b + 1], Xf[b:b + 1], fs_e=_FS_E, fs_f=_FS_F,
                                                  tmin_f=_TMIN_F, fps=_FPS, t_end=_TEND, lag_s=coupling["lag"])
    eeg = {name: _disp(eeg_s[name][0]).T.tolist() for name in eeg_s}          # {band: [T, ch_e]}
    fnirs = {"neural": _disp(neural_s[0]).T.tolist()}                          # [T, ch_f]
    cov = bc.coverage_map(pos_e, pos_f, _COV_GRID)                            # [g, g] locality confidence

    out = {
        "subject": args.subject, "block": args.block, "label": int(ye[b]),
        "classes": ["0-back", "2-back", "3-back"],
        "frame_times": t_dst.round(3).tolist(),
        "pos_eeg": _clean(pos_e).tolist(), "pos_fnirs": _clean(pos_f).tolist(),
        "eeg": eeg, "fnirs": fnirs, "coverage": cov.round(3).tolist(), "cov_grid": _COV_GRID,
        "coupling": {k: round(float(v), 2) for k, v in coupling.items()},   # derived lag/decay/beta (not fixed)
    }
    dst = REPO / "neuroviz" / "web" / "data" / f"brain_camera_subject{args.subject}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out))
    print(f"-> {dst.relative_to(REPO)}  ({len(t_dst)} frames, EEG {pos_e.shape[0]}ch, fNIRS {pos_f.shape[0]}ch, "
          f"class {out['label']}, derived lag {coupling['lag']:.1f}s decay {coupling['decay']:.1f}s "
          f"β {coupling['beta']:.2g})")


def _disp(x: np.ndarray) -> np.ndarray:
    """Per-channel-map display normalization: center + scale to ~[-1,1] over space+time (so the colormap uses
    the full range without one hot frame washing the rest out)."""
    x = x - np.median(x)
    return x / (np.percentile(np.abs(x), 98) + 1e-9)


def _clean(pos: np.ndarray) -> np.ndarray:
    """Replace non-finite positions with 0 (unused channels) so JSON is valid."""
    return np.where(np.isfinite(pos), pos, 0.0).round(4)


if __name__ == "__main__":
    main()
