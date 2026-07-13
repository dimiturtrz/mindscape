"""Fair test of the brain-camera FUSION features under the method that actually works for EEG workload.

The 3D-CNN eval was a strawman (overfits 702 samples). Here we take ONLY the new cross-modal feature — the
joint fused signal (EEG strength × co-located fNIRS CBSI × locality coverage), collapsed to EEG channel format
`[n, n_e, T]` — and run the exact decoder that gives the 0.580 EEG reference: spatial covariance -> per-subject
re-centered tangent space -> LR (`transfer.zero_shot_predict`). No raw EEG, no raw fNIRS — just the frankenstein.

    python -m neuroscan.tasks.workload.fusion_riemann_eval

Beat ~0.58 = the fused coupling carries something a single modality doesn't. ~chance/≤0.58 = a FAIR null (right
representation, right model, per-subject re-centered) — retire the brain-camera as a viz win, not a decode one.
"""
from __future__ import annotations

import logging

import numpy as np
from pyriemann.estimation import Covariances
from sklearn.model_selection import StratifiedGroupKFold

from baselines.eeg import transfer
from core.data import store
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.eeg.base import EpochCfg
from core.data.fnirs import shin2017 as fnmod
from core.data.fnirs.base import FnirsCfg
from core.features import fusion as bc
from neuroscan.evaluation import metrics

logger = logging.getLogger(__name__)

_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_FN_TMAX, _FPS, _TEND = 32.0, 10.0, 20.0
_SEEDS, _K = [0], 5          # 1 seed default (escalate-on-signal, bd); k-fold = validity not rigor
_CSD = True                                        # surface-Laplacian deblur of EEG before fusion
_FNIRS_BASELINE_ACC = 0.595                        # EEG-only re-centered Riemann reference (0.580) + margin


def _cov(X):
    return Covariances("oas").transform(X.astype(np.float64))


def _build_all(band="sum"):
    me = store.Store.load("shin2017_nback_eeg", _EEG_CFG)
    mf = store.Store.load("shin2017_nback", FnirsCfg(tmax=_FN_TMAX))
    subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
    ch_e = eegmod.Shin2017NbackEegAdapter.adapter().channels()
    pos_e = bc.EegMontage.eeg_positions(ch_e)
    Cs, ys, gs = [], [], []
    for s in subs:
        Xe, ye = store.Store.gather(me.filter(me["subject"] == s))
        Xf, yf = store.Store.gather(mf.filter(mf["subject"] == s))
        assert np.array_equal(ye, yf), f"subject {s} EEG/fNIRS misaligned"
        if _CSD:
            Xe = bc.CSD.csd_transform(Xe, ch_e, 100.0)                 # spatial deblur before fusion
        pos_f = bc.FnirsMontage.fnirs_positions(fnmod.Shin2017NirsAdapter.adapter()._subject_dir(int(s)))
        joint, _ = bc.BrainCamera.fused_node_series(bc.PairedModalities(Xe, Xf, pos_e, pos_f), band=band,
                                                    series=bc.SeriesConfig(fps=_FPS, t_end=_TEND))
        Cs.append(_cov(joint))
        ys.append(ye)
        gs.append(np.array([s] * len(ye)))
    return np.concatenate(Cs), np.concatenate(ys), np.concatenate(gs)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    C, y, g = _build_all()
    logger.info(f"fused-only riemann · {C.shape[0]} blocks · {len(set(g))} subj · cov {C.shape[1:]} · chance {1/(y.max()+1):.3f}")
    accs, kaps = [], []
    for seed in _SEEDS:
        for tr, te in StratifiedGroupKFold(_K, shuffle=True, random_state=seed).split(C, y, g):
            # winning EEG method: per-subject re-center (train AND test, unsupervised) -> tangent -> LR
            proba = transfer.zero_shot_predict(transfer.Domain(C[tr], y[tr], g[tr]),
                                               transfer.Domain(C[te], groups=g[te]), scale=False)
            pred = proba.argmax(1)
            accs.append(metrics.accuracy(y[te], pred))
            kaps.append(metrics.kappa(y[te], pred))
    a, k = float(np.mean(accs)), float(np.mean(kaps))
    logger.info(f"\n  fused-only (joint EEG×fNIRS×coverage) · re-centered tangent · cross-subject {len(_SEEDS)}x{_K}-fold: "
          f"acc {a:.3f} ± {np.std(accs):.3f} · κ {k:.3f}")
    logger.info("  reference (same protocol, EEG-only re-centered Riemann): best-single 0.580")
    logger.info(f"  Δ vs best-single: {a - 0.580:+.3f}  ->  {'FUSION CASHED something' if a > _FNIRS_BASELINE_ACC else 'fair null (fusion adds nothing decodable)'}")


if __name__ == "__main__":
    main()
