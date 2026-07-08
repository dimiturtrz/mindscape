"""Cross-dataset EEG1<->EEG2 bridge — pure set/label reconciliation, synthetic concept names."""
import numpy as np
import pytest

from neuroscan.evaluation.cross_dataset import align_channels, common_channel_order, holdout_mask


def test_holdout_mask_excludes_eval_concepts_from_training():
    names = np.array(["aardvark", "antelope", "axe", "antelope"])
    mask = holdout_mask(names, {"antelope"})            # antelope is an eval concept -> drop both its trials
    assert list(mask) == [True, False, True, False]


def test_common_channel_order_keeps_a_order_intersection():
    # EEG1-like (has Fz not Cz) vs EEG2-like (has Cz not Fz); common = the shared ones in names_a order
    a = ["Fp1", "Fz", "F3", "Cz_missing_here", "O1"]
    b = ["O1", "F3", "Fp1", "Cz"]                        # no Fz, no 'Cz_missing_here'
    assert common_channel_order(a, b) == ["Fp1", "F3", "O1"]   # a's order, only those also in b


def test_align_channels_reorders_by_name():
    # eeg channels in src order; align to a target order (a permutation + drop)
    src = ["A", "B", "C", "D"]
    eeg = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype(float)   # [n=2, C=4, t=3]
    out = align_channels(eeg, src, ["C", "A"])          # keep C then A, drop B,D
    assert out.shape == (2, 2, 3)
    assert np.array_equal(out[:, 0], eeg[:, 2]) and np.array_equal(out[:, 1], eeg[:, 0])


def test_align_channels_errors_on_missing_target():
    eeg = np.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="missing"):
        align_channels(eeg, ["A", "B"], ["A", "Z"])     # Z not in source
