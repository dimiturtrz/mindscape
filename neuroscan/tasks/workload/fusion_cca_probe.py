"""Cheapest linear test of si7: is there a SHARED EEG↔fNIRS latent that decodes workload better than either
modality alone — i.e. does INFERRING a common neural cause beat MULTIPLYING the observables?

The three prior fusion nulls all combined the raw observations: output-space multiplicative joint (060),
plain source projection (728), fNIRS-informed inverse (4so). This probes a DIFFERENT axis — the shared
*subspace*. CCA finds the directions of maximal EEG↔fNIRS correlation (the linear common latent); we decode
workload from that shared estimate and compare, in ONE matched fold loop, against EEG-alone, fNIRS-alone, and
naive feature-concat. Features reuse the strong decoders' own views: EEG = per-subject re-centered tangent
(the 0.58 Riemann decoder's feature space), fNIRS = the full descriptor bank.

    python -m neuroscan.tasks.workload.fusion_cca_probe

Read: `shared` > max(eeg, fnirs) beyond fold noise = the common cause is MORE discriminative than either raw
modality (denoising via shared inference cashes) → si7 generative build is worth it. `shared` ≈/< the best
single modality = the linear shared latent carries no extra workload signal; a generative si7 would have to
find NON-linear shared structure (higher bar, not fatal — CCA is linear). Cheap gate, single seed/fold-set,
no hardening (owner: fast keep/kill on the number + mechanism, not multi-seed).
"""
from __future__ import annotations

import logging

import numpy as np
from jaxtyping import Float, Int
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from baselines.eeg import transfer
from core.data import store
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from core.features import DescriptorBank
from core.features import fusion as bc
from neuroscan.evaluation import metrics
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload.riemann import Riemann

logger = logging.getLogger(__name__)


class FusionCcaProbe:
    """EEG↔fNIRS shared-latent (CCA) workload decode vs single-modality + naive-concat, one matched fold loop."""

    _EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)   # matches fusion_riemann_eval
    _FN_TMAX = 32.0
    _CSD = True                    # surface-Laplacian deblur of EEG before covariance (matches the 0.58 recipe)
    _SEEDS, _K = [0], 5            # single seed default (fast keep/kill; k-fold = validity not rigor)
    _N_COMP = 10                   # CCA canonical components (shared-subspace dim)
    _EEG_FS = 100.0

    @classmethod
    def _load(cls) -> tuple[Float[np.ndarray, "n ce te"], Float[np.ndarray, "n cf tf"],
                            Int[np.ndarray, "n"], Int[np.ndarray, "n"]]:
        """Block-aligned EEG+fNIRS over the subjects both modalities share -> (Xe[N,ch,t], Xf[N,72,t], y, g).
        `g` = per-block subject index, for grouped cross-subject folds + per-subject EEG re-centering."""
        me = store.Store.load("shin2017_nback_eeg", cls._EEG_CFG)
        mf = store.Store.load("shin2017_nback", FnirsCfg(tmax=cls._FN_TMAX))
        subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
        ch_e = eegmod.Shin2017NbackEegAdapter.adapter().channels()
        xes: list[np.ndarray] = []
        xfs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        gs: list[np.ndarray] = []
        for i, s in enumerate(subs):
            xe, xf, ye = store.Store.gather_aligned(me, mf, s)
            if cls._CSD:
                xe = bc.CSD.csd_transform(xe, ch_e, cls._EEG_FS)
            xes.append(xe)
            xfs.append(xf)
            ys.append(ye)
            gs.append(np.full(len(ye), i))
        return np.concatenate(xes), np.concatenate(xfs), np.concatenate(ys), np.concatenate(gs)

    @classmethod
    def _features(cls, xe: Float[np.ndarray, "n ce te"], xf: Float[np.ndarray, "n cf tf"],
                  g: Int[np.ndarray, "n"]) -> tuple[Float[np.ndarray, "n de"], Float[np.ndarray, "n df"]]:
        """EEG re-centered-tangent feature (the strong decoder's feature space, per-subject centred) and the
        full fNIRS descriptor bank. Both are leakage-free per block (EEG re-centering is per-subject unsupervised;
        the bank is per-block), so they can be built once over all blocks before the fold split."""
        eeg = transfer.recentered_tangent_features(Riemann.cov(xe), g)          # [N, d(d+1)/2]
        fnirs, _ = DescriptorBank.extract_bank(xf)                              # [N, 72*K]
        return eeg, fnirs

    @staticmethod
    def _decode(feat_tr: Float[np.ndarray, "m d"], feat_te: Float[np.ndarray, "k d"],
                ytr: Int[np.ndarray, "m"], yte: Int[np.ndarray, "k"]) -> float:
        """Shrinkage-LDA fit on train features, accuracy on test — the shared classifier every arm uses, so the
        arms differ ONLY in their feature set (matched comparison)."""
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(feat_tr, ytr)
        return metrics.Metrics.accuracy(yte, lda.predict(feat_te))

    @classmethod
    def _fold_arms(cls, eeg: Float[np.ndarray, "n de"], fnirs: Float[np.ndarray, "n df"],
                   y: Int[np.ndarray, "n"], tr: Int[np.ndarray, "m"], te: Int[np.ndarray, "k"]
                   ) -> dict[str, float]:
        """One fold: standardise both modalities on train, fit CCA on train to get the shared subspace, decode
        four arms (eeg / fnirs / naive-concat / cca-shared) with the same LDA. `shared` = the mean of the two
        aligned canonical projections — both estimate the same latent, averaging denoises."""
        se, sf = StandardScaler().fit(eeg[tr]), StandardScaler().fit(fnirs[tr])
        ee_tr, ee_te = se.transform(eeg[tr]), se.transform(eeg[te])
        ef_tr, ef_te = sf.transform(fnirs[tr]), sf.transform(fnirs[te])

        cca = CCA(n_components=cls._N_COMP).fit(ee_tr, ef_tr)
        ue_tr, uf_tr = cca.transform(ee_tr, ef_tr)
        ue_te, uf_te = cca.transform(ee_te, ef_te)
        shared_tr, shared_te = (ue_tr + uf_tr) / 2, (ue_te + uf_te) / 2

        ytr, yte = y[tr], y[te]
        return {
            "eeg": cls._decode(ee_tr, ee_te, ytr, yte),
            "fnirs": cls._decode(ef_tr, ef_te, ytr, yte),
            "concat": cls._decode(np.hstack([ee_tr, ef_tr]), np.hstack([ee_te, ef_te]), ytr, yte),
            "shared": cls._decode(shared_tr, shared_te, ytr, yte),
        }

    @classmethod
    def main(cls) -> None:
        Cli.setup_logging()
        xe, xf, y, g = cls._load()
        eeg, fnirs = cls._features(xe, xf, g)
        logger.info(f"fusion-CCA probe · {len(y)} blocks · {len(np.unique(g))} subj · eeg {eeg.shape[1]}d · "
                    f"fnirs {fnirs.shape[1]}d · cca {cls._N_COMP} comp · chance {1 / (y.max() + 1):.3f}")

        arms: dict[str, list[float]] = {"eeg": [], "fnirs": [], "concat": [], "shared": []}
        for seed in cls._SEEDS:
            for tr, te in StratifiedGroupKFold(cls._K, shuffle=True, random_state=seed).split(eeg, y, g):
                for name, acc in cls._fold_arms(eeg, fnirs, y, tr, te).items():
                    arms[name].append(acc)

        stats = {name: (float(np.mean(accs)), float(np.std(accs))) for name, accs in arms.items()}
        logger.info(f"\n  cross-subject {len(cls._SEEDS)}x{cls._K}-fold workload decode (shrinkage-LDA):")
        for name in ("eeg", "fnirs", "concat", "shared"):
            m, s = stats[name]
            logger.info(f"    {name:<8} {m:.3f} ± {s:.3f}")

        best_single = max(stats["eeg"][0], stats["fnirs"][0])
        delta = stats["shared"][0] - best_single
        spark = delta > stats["shared"][1]     # beats the best single modality beyond this arm's own fold-std
        verdict = ("SHARED-LATENT CASHES — build generative si7" if spark
                   else "null: linear shared latent no better than best single modality (generative bar higher)")
        logger.info(f"\n  Δ shared − best-single ({'eeg' if stats['eeg'][0] >= stats['fnirs'][0] else 'fnirs'}): "
                    f"{delta:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    FusionCcaProbe.main()
