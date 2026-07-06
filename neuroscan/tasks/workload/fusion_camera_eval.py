"""Does the brain-camera spatiotemporal fusion cash the oracle headroom flat fusion couldn't?

Builds the fused [C,H,W,T] EEG+fNIRS surface-video per block, trains a tiny 3D-CNN, scores cross-subject
(5-fold GroupKFold — matched to run_fusion so it's comparable). Reference (same protocol, run_fusion):
best-single 0.580 · late 0.587 · feature 0.564 · ORACLE 0.752. Beat ~0.59 = the geometry+time representation
cashed something flat fusion destroyed; ~0.58 = another honest null (but now on the RIGHT representation).

    python -m neuroscan.tasks.workload.fusion_camera_eval
"""
from __future__ import annotations

import numpy as np

from baselines.fusion.spatiotemporal import BrainCameraNet
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.fnirs.base import FnirsCfg
from core.data.fnirs import shin2017 as fnmod
from core.features import brain_camera as bc
from neuroscan.evaluation import metrics

_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_GRID, _FPS, _TEND = 16, 10.0, 20.0                # hemodynamic lag derived per subject (no fixed shift)
_SEEDS, _K = [0, 1, 2], 5


def _build_all():
    me = store.load("shin2017_nback_eeg", _EEG_CFG)
    mf = store.load("shin2017_nback", FnirsCfg(tmax=32.0))     # past _TEND so read-forward (τ+lag) fills the tail
    subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
    pos_e = bc.eeg_positions(eegmod.adapter().channels())
    Xs, ys, gs = [], [], []
    for s in subs:
        Xe, ye = store.gather(me.filter(me["subject"] == s))
        Xf, yf = store.gather(mf.filter(mf["subject"] == s))
        assert np.array_equal(ye, yf), f"subject {s} EEG/fNIRS misaligned"
        pos_f = bc.fnirs_positions(fnmod.adapter()._subject_dir(int(s)))
        Xs.append(bc.build_tensor(Xe, Xf, pos_e, pos_f, grid=_GRID, fps=_FPS, t_end=_TEND))
        ys.append(ye); gs.append(np.array([s] * len(ye)))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    X, y, g = _build_all()
    print(f"brain-camera fusion · {X.shape[0]} blocks · {len(set(g))} subj · tensor {X.shape[1:]} · "
          f"grid {_GRID} fps {_FPS} lag derived/subj · chance {1/(y.max()+1):.3f}")
    accs, kaps = [], []
    for seed in _SEEDS:
        for tr, te in StratifiedGroupKFold(_K, shuffle=True, random_state=seed).split(X, y, g):
            clf = BrainCameraNet(n_classes=int(y.max()) + 1, seed=seed).fit(X[tr], y[tr])
            pred = clf.predict_proba(X[te]).argmax(1)
            accs.append(metrics.accuracy(y[te], pred)); kaps.append(metrics.kappa(y[te], pred))
    a, k = float(np.mean(accs)), float(np.mean(kaps))
    print(f"\n  brain-camera 3D-CNN · cross-subject {len(_SEEDS)}x{_K}-fold: acc {a:.3f} ± {np.std(accs):.3f} · κ {k:.3f}")
    print(f"  reference (per-subject-z features -> LDA): best-single 0.580 · late 0.587 · feature 0.564 · oracle 0.752")
    print(f"  Δ vs best-single: {a - 0.580:+.3f}  ->  {'CASHED something' if a > 0.595 else 'null'}")
    print("  NOTE: not a fair representation test — this is a raw 3D-CNN (no per-subject re-centering) on 702 "
          "cross-subject samples (overfits); the 0.580 ref had per-subject-z + LDA. Fair test = re-center + a "
          "readout that doesn't overfit. The null is the METHOD, not proof the representation is empty.")


if __name__ == "__main__":
    main()
