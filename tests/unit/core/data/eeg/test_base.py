"""core.data.eeg.base — the EpochCfg cache-key recipe + the MoabbMIAdapter interface wiring.

Pure logic only, no MOABB download: EpochCfg.key must encode every recipe param (two recipes never collide,
None tmax renders as "full"), the canonical MI label map is the fixed 4-class convention, and MoabbMIAdapter
defaults its label_map to CANONICAL_MI while keeping the injected montage/subjects contract (subjects() reads
the dataset's subject_list). get_data (the actual paradigm epoching) needs a real dataset -> not unit-tested.
"""
from core.data.eeg.base import CANONICAL_MI, EpochCfg, MoabbMIAdapter


def test_canonical_mi_is_the_fixed_four_class_convention():
    assert CANONICAL_MI == {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}


def test_epochcfg_key_encodes_recipe_and_disambiguates():
    base = EpochCfg()                                    # 8-32 Hz, t0-full, r128
    assert base.key() == "b8p0-32p0_t0p0-full_r128p0"    # dots -> 'p', None tmax -> 'full'
    assert EpochCfg(fmin=4.0).key() != base.key()        # a changed param yields a different cache dir
    assert EpochCfg(tmax=2.0).key().endswith("t0p0-2p0_r128p0")   # concrete tmax rendered, not 'full'


class _DummyDataset:
    subject_list = [1, 2, 3]


def test_adapter_defaults_label_map_to_canonical():
    a = MoabbMIAdapter("bnci", _DummyDataset)
    assert a.label_map == CANONICAL_MI
    assert a.name == "bnci" and a.n_classes == 4
    assert a.subjects() == [1, 2, 3]                     # reads the dataset's subject_list, no download


def test_adapter_keeps_injected_label_map():
    custom = {"left_hand": 0, "right_hand": 1}
    a = MoabbMIAdapter("two", _DummyDataset, n_classes=2, label_map=custom)
    assert a.label_map == custom and a.n_classes == 2
