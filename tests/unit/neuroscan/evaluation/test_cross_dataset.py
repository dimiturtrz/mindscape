"""Cross-dataset EEG1<->EEG2 bridge — pure set/label reconciliation, synthetic concept names."""
import numpy as np

from neuroscan.evaluation.cross_dataset import align_targets, holdout_mask, name_to_bank_index


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
