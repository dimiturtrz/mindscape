"""Cross-dataset EEG1<->EEG2 bridge — pure set/label reconciliation, synthetic concept names."""
import numpy as np
import pytest

from neuroscan.evaluation.cross_dataset import (
    align_channels,
    align_targets,
    common_channel_order,
    holdout_mask,
    name_to_bank_index,
)


def test_holdout_mask_excludes_eval_concepts_from_training():
    names = np.array(["aardvark", "antelope", "axe", "antelope"])
    mask = holdout_mask(names, {"antelope"})            # antelope is an eval concept -> drop both its trials
    assert list(mask) == [True, False, True, False]


def test_name_to_bank_index_orders_by_candidate_list():
    idx = name_to_bank_index(["antelope", "aircraft_carrier", "axe"])
    assert idx == {"antelope": 0, "aircraft_carrier": 1, "axe": 2}


def test_align_targets_maps_names_and_drops_out_of_bank():
    bank = name_to_bank_index(["antelope", "axe"])       # candidate bank has 2 concepts
    names = np.array(["axe", "aardvark", "antelope"])     # 'aardvark' not in the bank -> dropped
    labels, keep = align_targets(names, bank)
    assert list(keep) == [True, False, True]
    assert list(labels) == [1, 0]                        # axe->1, antelope->0, aligned to kept trials in order


def test_align_targets_all_present():
    bank = name_to_bank_index(["a", "b", "c"])
    labels, keep = align_targets(np.array(["c", "a", "b", "b"]), bank)
    assert keep.all() and list(labels) == [2, 0, 1, 1]


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
