"""Is the EEG->blood hemodynamic offset derivable from our data? — the provenance for the fusion lag constant.

We want to replace the hardcoded 5 s fNIRS lag with a value DERIVED from the data (no magic numbers). This probe
tests at what granularity that derivation is stable:
  - per-subject / per-block (what channel_series would do live), and
  - POOLED over all paired subjects.
It scans the whole-head-mean EEG-β envelope vs fNIRS CBSI correlation as a function of pure shift, and runs the
gamma-kernel `estimate_coupling` fit, so we can SEE whether there's a real coupling peak or just noise.

    python -m neuroscan.tasks.workload.coupling_probe

Finding (2026-07-06, 26 paired subjects): per-subject is unusable (lag scatters 1-12 s, |β|~0 — 9 blocks × 20 s
is too little). Pooled shows a real but WEAK, NEGATIVE coupling — whole-head β-power anti-correlates with CBSI
(~HbO) at ~7-8 s: event-related desync (power drops on activation) trailed by the HbO rise. So the offset is only
derivable at the POPULATION level; use the pooled value as the argued constant, not a fixed 5 s.
"""
from __future__ import annotations

import logging

import numpy as np

from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from core.features import fusion as bc
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)

_FS_E, _FS_F, _TMIN_F, _FPS, _TEND = 100.0, 10.0, -2.0, 10.0, 20.0
_BETA = (13.0, 30.0)                                 # β power ~ the (de)synchronization that couples to blood
_LAG_STABLE_STD_S = 2                                # per-subject lag std (s) below which the coupling fit is STABLE


class CouplingProbe:
    """EEG->blood coupling-lag probe helpers — the free helpers folded in as staticmethods."""

    @staticmethod
    def _global_series(subject_frames):
        """Whole-head-mean EEG-β envelope + zero-lag fNIRS CBSI per block, on the shared grid -> `[n, T]` each."""
        t_dst = np.arange(0, _TEND, 1.0 / _FPS)
        drives, resps, groups = [], [], []
        for s, (Xe, Xf) in subject_frames:
            ch_f = Xf.shape[1] // 2
            te = np.arange(Xe.shape[2]) / _FS_E
            beta = bc.Series._resample_time(bc.Series._band_env(Xe, _FS_E, _BETA), te, t_dst).mean(1)   # [n, T]
            tf = _TMIN_F + np.arange(Xf.shape[2]) / _FS_F
            cbsi = bc.Chromophore.cbsi_neural(bc.Series._resample_time(Xf[:, :ch_f, :], tf, t_dst),
                                              bc.Series._resample_time(Xf[:, ch_f:, :], tf, t_dst)).mean(1)   # [n, T]
            drives.append(beta)
            resps.append(cbsi)
            groups.append([s] * len(beta))
        return drives, resps, groups


def main():
    Cli.setup_logging()
    me = store.Store.load("shin2017_nback_eeg", EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=_FS_E))
    mf = store.Store.load("shin2017_nback", FnirsCfg())
    subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
    frames = [(s, (store.Store.gather(me.filter(me["subject"] == s))[0],
                   store.Store.gather(mf.filter(mf["subject"] == s))[0]))
              for s in subs]
    drives, resps, _ = CouplingProbe._global_series(frames)
    D, R = np.concatenate(drives), np.concatenate(resps)
    logger.info(f"pooled n={D.shape[0]} blocks · {len(subs)} subjects")

    # pure-shift scan: signed whole-head correlation vs lag — is there a coherent peak?
    Rz = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-9)
    logger.info("shift(s) : mean signed corr")
    for shift_s in range(0, 12):
        lag = round(shift_s * _FPS)
        drive_shifted = np.roll(D, lag, axis=1)
        if lag > 0:
            drive_shifted[:, :lag] = D[:, :1]
        drive_z = (drive_shifted - drive_shifted.mean(1, keepdims=True)) / (drive_shifted.std(1, keepdims=True) + 1e-9)
        logger.info(f"  {shift_s:2d}   {float((drive_z * Rz).mean(1).mean()):+.3f}")

    lag, decay, beta = bc.Coupling.estimate_coupling(D, R, _FPS)
    logger.info(f"\nPOOLED gamma fit: lag {lag:.1f}s · decay {decay:.2f}s · β {beta:.2g}")
    per = np.array([bc.Coupling.estimate_coupling(dv, rp, _FPS)[0] for dv, rp in zip(drives, resps, strict=True)])
    logger.info(f"per-subject lag: mean {per.mean():.1f}s · std {per.std():.1f}s · "
          f"range [{per.min():.1f}, {per.max():.1f}] "
          f"-> {'STABLE' if per.std() < _LAG_STABLE_STD_S else 'UNSTABLE (pool instead)'}")


if __name__ == "__main__":
    main()
