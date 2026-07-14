"""Cross-dataset EEG->image bridge — reconcile THINGS-EEG1 and THINGS-EEG2 for a zero-shot transfer test.

Both datasets show the SAME 1,854 THINGS concepts, so an encoder trained on one can be evaluated on the other
— the hardest leakage-free generalization test (different people, different rig, same semantics). Two reconciliations
make it leakage-free, and both are pure set/label logic (no EEG, no GPU) so they're unit-tested here:

  1. Concept identity is the THINGS NAME (EEG1 carries it inline; EEG2 derives it from its concept order). The
     retrieval target for a trial is that concept's shared CLIP prototype — looked up by name, so the two
     datasets land in one embedding space.
  2. Zero-shot holdout: EEG2's 200 test concepts also occur in EEG1, so training the EEG1 encoder on them would
     leak the eval concepts. `holdout_mask` removes them from the training side; retrieval still runs on the
     200-concept bank. The result is cross-dataset AND concept-zero-shot AND cross-subject at once.
"""
from __future__ import annotations

import numpy as np
from jaxtyping import Bool, Float, Shaped


class CrossDataset:
    @staticmethod
    def holdout_mask(concept_names: Shaped[np.ndarray, "n"], holdout: set[str]) -> Bool[np.ndarray, "n"]:
        """Boolean mask of trials to KEEP for training: every trial whose concept is NOT in the eval `holdout`
        set. Keeps the cross-dataset encoder from ever seeing the concepts it will be retrieved on."""
        names = np.asarray(concept_names)
        return np.array([name not in holdout for name in names], dtype=bool)

    @staticmethod
    def common_channel_order(names_a: list[str], names_b: list[str]) -> list[str]:
        """The electrodes present in BOTH montages, in `names_a`'s order — the shared spatial layout a
        cross-dataset encoder must use. THINGS-EEG1 and -EEG2 share 62 of 63 electrodes (EEG1 has Fz not Cz,
        EEG2 has Cz not Fz) but in different channel ORDER, so without this the encoder's spatial filters land
        on the wrong electrodes at eval — the confound that pins cross-dataset retrieval to chance."""
        in_b = set(names_b)
        return [name for name in names_a if name in in_b]

    @staticmethod
    def align_channels(eeg: Float[np.ndarray, "n ch t"], src_names: list[str],
                       target_names: list[str]) -> Float[np.ndarray, "n ch_out t"]:
        """Reindex `eeg` [n, C, t] so its channel axis matches `target_names` BY NAME (drop channels not in the
        target; error if the target names a channel absent from the source). Makes two datasets' differently-
        ordered montages line up before an encoder trained on one is applied to the other."""
        src_index = {name: i for i, name in enumerate(src_names)}
        missing = [name for name in target_names if name not in src_index]
        if missing:
            raise ValueError(f"source channels missing {missing} — cannot align to target montage")
        order = [src_index[name] for name in target_names]
        return eeg[:, order, :]
