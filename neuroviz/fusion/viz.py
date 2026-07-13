"""Brain-camera visualization — render the fused EEG+fNIRS surface-video as an animated GIF.

Layout (per the design): LEFT column = the two raw inputs stacked (EEG band-power map on top, fNIRS HbO map
below); RIGHT = the fused heatmap (EEG + fNIRS overlaid). Sensor nodes dotted on each. One block, animated
over time — the "watch the brain fire" view, before deciding feature extraction or committing to the web build.

    python -m neuroviz.fusion.viz --subject 1 --block 0 --band alpha
"""
from __future__ import annotations

import argparse

import numpy as np


def _maps(subject: int, block: int, band: str):
    from core.data import store
    from core.data.eeg.base import EpochCfg
    from core.data.eeg import shin2017_nback_eeg as eegmod
    from core.data.fnirs.base import FnirsCfg
    from core.data.fnirs import shin2017 as fnmod
    from core.features import fusion as bc

    me = store.load("shin2017_nback_eeg", EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0))
    mf = store.load("shin2017_nback", FnirsCfg(tmax=32.0))     # past 20 s so the read-forward tail has blood
    s = str(subject)
    Xe, ye = store.gather(me.filter(me["subject"] == s))
    Xf, _ = store.gather(mf.filter(mf["subject"] == s))
    Xe = bc.CSD.csd_transform(Xe, eegmod.adapter().channels(), 100.0)  # surface-Laplacian deblur (match export)
    pos_e = bc.EegMontage.eeg_positions(eegmod.adapter().channels())
    pos_f = bc.FnirsMontage.fnirs_positions(fnmod.adapter()._subject_dir(subject))
    X = bc.BrainCamera.build_tensor(bc.PairedModalities(Xe, Xf, pos_e, pos_f), grid=16,
                                    series=bc.SeriesConfig(fps=10.0, t_end=20.0))  # [n,C=5,16,16,T], lag derived
    band_idx = {"theta": 0, "alpha": 1, "beta": 2}[band]
    eeg = X[block, band_idx]            # [16,16,T] EEG band-power map
    neural = X[block, 3]               # [16,16,T] fNIRS CBSI neural map (lag-aligned; ch 3, ch 4 = coverage)
    return eeg, neural, pos_e, pos_f, int(ye[block])


def _grid_nodes(pos, grid=16):
    """Map unit-disk positions [-1,1] to grid pixel coords for overlay."""
    ok = np.isfinite(pos).all(1)
    xy = (pos[ok] + 1) / 2 * (grid - 1)
    return xy[:, 0], xy[:, 1]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--band", default="alpha", choices=["theta", "alpha", "beta"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=4, help="frame subsample for a lighter gif")
    args = ap.parse_args()

    eeg, hbo, pos_e, pos_f, label = _maps(args.subject, args.block, args.band)
    T = eeg.shape[2]
    frames = range(0, T, args.stride)
    exe, eye = _grid_nodes(pos_e)
    fxe, fye = _grid_nodes(pos_f)

    fig, ax = plt.subplots(1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [1, 1.15]})
    # left = stacked raw inputs (2 sub-rows); right = fused
    gsL = fig.add_gridspec(2, 1, left=0.05, right=0.47, top=0.88, bottom=0.08, hspace=0.25)
    ax[0].remove(); ax[1].set_position([0.55, 0.12, 0.4, 0.72])
    axE = fig.add_subplot(gsL[0]); axF = fig.add_subplot(gsL[1]); axFused = ax[1]
    for a, t in [(axE, f"EEG {args.band} power"), (axF, "fNIRS CBSI neural (lag derived)"), (axFused, "FUSED")]:
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"brain-camera · subj {args.subject} · block {args.block} · class {label} (0-back/2-back/3-back)",
                 fontsize=11)

    # robust display range (z-scored maps have outliers — min/max washes them flat; use 2/98th percentiles,
    # symmetric for the diverging CBSI map) so the structure shows, matching the web view's _disp scaling
    elo, ehi = np.percentile(eeg, [2, 98])
    fh = np.percentile(np.abs(hbo), 98) + 1e-9
    imE = axE.imshow(eeg[:, :, 0], cmap="magma", vmin=elo, vmax=ehi, origin="lower")
    imF = axF.imshow(hbo[:, :, 0], cmap="RdBu_r", vmin=-fh, vmax=fh, origin="lower")
    # fused: EEG magma base + fNIRS as a translucent RdBu overlay
    imFa = axFused.imshow(eeg[:, :, 0], cmap="magma", vmin=elo, vmax=ehi, origin="lower")
    imFb = axFused.imshow(hbo[:, :, 0], cmap="RdBu_r", vmin=-fh, vmax=fh, origin="lower", alpha=0.45)
    for a, (nx, ny) in [(axE, (exe, eye)), (axF, (fxe, fye)), (axFused, (exe, eye))]:
        a.scatter(nx, ny, s=8, c="cyan", edgecolors="k", linewidths=0.3)   # node overlay
    axFused.scatter(fxe, fye, s=8, c="lime", edgecolors="k", linewidths=0.3)

    def update(t):
        imE.set_data(eeg[:, :, t]); imF.set_data(hbo[:, :, t])
        imFa.set_data(eeg[:, :, t]); imFb.set_data(hbo[:, :, t])
        return imE, imF, imFa, imFb

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    out = args.out or f"brain_camera_s{args.subject}_b{args.block}_{args.band}.gif"
    anim.save(out, writer="pillow", fps=8)
    print(f"saved -> {out}  ({len(list(frames))} frames)")


if __name__ == "__main__":
    main()
