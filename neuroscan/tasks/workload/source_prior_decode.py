"""fNIRS-informed source-space fusion decode (bd 4so) — does regularizing the EEG inverse with the fNIRS
'where' beat a plain inverse for cross-subject workload decode?

The physically-correct complementarity (Liu, Belliveau & Dale 1998): the fNIRS activation map becomes a
per-source prior variance on the ill-posed EEG minimum-norm inverse (`weighted_min_norm_inverse`), drawing the
cortical estimate toward fNIRS-active patches. This is the consumer the 4so operator was stripped for lacking.
Shin n-back has paired EEG+fNIRS on the same task/subjects, so each subject's OWN fNIRS supplies its prior —
unsupervised (per-channel HbO response magnitude, no labels), interpolated from the fNIRS optode disk onto the
source-space vertices.

Four matched arms, cross-subject re-centered-tangent + LR (the winning EEG method), 68-DK-parcel frame:
  sensor          28-ch covariance                        (ref ~0.594)
  dSPM source     plain minimum-norm parcels (728)         (ref ~0.602)
  uniform prior   weighted-min-norm, w=1 (control)         isolates the K + parcel-aggregation change
  fNIRS prior     weighted-min-norm, w=fNIRS activation    the treatment — does 'where' add decodable info?

Beat uniform-prior AND dSPM = the fNIRS spatial prior cashes complementarity in source space. ≤ them = a fair
null (the prior regularizes, doesn't inform the discriminant) — complementarity not decode-accessible even here.

    python -m neuroscan.tasks.workload.source_prior_decode
"""
from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from jaxtyping import Float

from core.data import store
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.eeg.base import EpochCfg
from core.data.fnirs import shin2017 as fnmod
from core.data.fnirs.base import FnirsCfg
from core.features.eeg.montage import EegMontage
from core.features.eeg.source import Source
from core.features.fnirs.montage import FnirsMontage
from core.features.fusion.source_prior import SourcePrior
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload.riemann import Riemann

logger = logging.getLogger(__name__)

_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_SFREQ = 100.0
_SEEDS, _K = [0], 5          # 1 seed default (escalate-on-signal, bd); k-fold = validity not rigor
_RBF_SIGMA = 0.15           # fNIRS→source interpolation width (unit-disk units)
_PRIOR_FLOOR = 0.1          # nonzero prior everywhere so EEG can still place sources fNIRS missed
_WIN = 0.01                 # min Δacc over the controls to call the fNIRS prior a real decode gain


class SourcePriorDecode:
    """fNIRS-informed source-space fusion decode helpers (bd 4so) — the free helpers folded in as staticmethods."""

    @classmethod
    def _fnirs_prior(cls, x_fnirs: Float[np.ndarray, "n ch_f t"], subject_dir: Any,
                     src2d: Float[np.ndarray, "src 2"]) -> Float[np.ndarray, "src"]:
        """Per-source prior `w [n_src]` from a subject's fNIRS: per-channel HbO response magnitude (std over time,
        mean over epochs — unsupervised) RBF-interpolated from the optode disk onto the source-space vertices."""
        act = np.asarray(x_fnirs[:, :36, :], dtype=np.float64).std(axis=-1).mean(axis=0)   # [36] HbO channels
        act = act / (act.max() + 1e-12)
        pos_f = FnirsMontage.fnirs_positions(subject_dir)                                   # [36, 2] unit disk
        d2 = ((src2d[:, None, :] - pos_f[None, :, :]) ** 2).sum(-1)                          # [n_src, 36]
        wgt = np.exp(-d2 / (2 * _RBF_SIGMA ** 2))
        a_src = (wgt @ act) / (wgt.sum(1) + 1e-12)                                           # [n_src]
        a_src = a_src / (a_src.max() + 1e-12)
        return _PRIOR_FLOOR + (1.0 - _PRIOR_FLOOR) * a_src

    @classmethod
    def _build(cls):
        """Per-subject covariances for the four arms + labels/groups, over the EEG∩fNIRS subjects."""
        me = store.Store.load("shin2017_nback_eeg", _EEG_CFG)
        mf = store.Store.load("shin2017_nback", cast(EpochCfg, FnirsCfg()))
        subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
        ch_e = eegmod.Shin2017NbackEegAdapter.adapter().channels()
        g, agg = SourcePrior.prior_leadfield(ch_e, _SFREQ)
        src = Source(ch_e, _SFREQ)
        src2d = EegMontage.to_unit_disk(src.source_positions()[:, :2])       # source flatmap
        arms: dict[str, list[np.ndarray]] = {"sensor": [], "dSPM": [], "uniform": [], "fNIRS": []}
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []
        for s in subs:
            xe, xf, ye = store.Store.gather_aligned(me, mf, s)
            w = cls._fnirs_prior(xf, fnmod.Shin2017NirsAdapter.adapter().subject_dir(int(s)), src2d)
            arms["sensor"].append(Riemann.cov(xe))
            arms["dSPM"].append(Riemann.cov(src.to_parcels(xe)))
            arms["uniform"].append(Riemann.cov(SourcePrior.parcels_from_leadfield(xe, g, agg, None)))
            arms["fNIRS"].append(Riemann.cov(SourcePrior.parcels_from_leadfield(xe, g, agg, w)))
            ys.append(ye)
            gs.append(np.array([s] * len(ye)))
            logger.info(f"  subject {s}: {len(ye)} blocks · prior w∈[{w.min():.2f},{w.max():.2f}]")
        return ({k: np.concatenate(v) for k, v in arms.items()}, np.concatenate(ys), np.concatenate(gs))

    @classmethod
    def main(cls):
        Cli.setup_logging()
        arms, y, g = cls._build()
        logger.info(f"\n{len(y)} blocks · {len(set(g.tolist()))} subj · chance {1 / (y.max() + 1):.3f} "
              f"· {len(_SEEDS)}x{_K}-fold re-centered tangent")
        res = {name: Riemann.cross_subject_decode(c, y, g, _SEEDS, _K) for name, c in arms.items()}
        for name in ("sensor", "dSPM", "uniform", "fNIRS"):
            a, sd = res[name]
            logger.info(f"  {name:14s} acc {a:.3f} ± {sd:.3f}")
        d_uni = res["fNIRS"][0] - res["uniform"][0]
        d_dspm = res["fNIRS"][0] - res["dSPM"][0]
        cashes = d_uni > _WIN and d_dspm > _WIN
        verdict = ("fNIRS PRIOR CASHES structure" if cashes else
                   "fair null (prior regularizes, doesn't inform the discriminant)")
        logger.info(f"  Δ fNIRS − uniform: {d_uni:+.3f} · Δ fNIRS − dSPM: {d_dspm:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    SourcePriorDecode.main()
