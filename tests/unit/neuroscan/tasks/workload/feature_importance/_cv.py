"""neuroscan.tasks.workload.feature_importance._cv — the shared subject-grouped fold generator.

Pure CV plumbing: Cv.grouped_folds yields one (train_idx, test_idx) per fold per seed (len(seeds)*k folds),
every split is subject-DISJOINT (no within-subject leakage), and across the k folds of a seed every sample is
tested exactly once. Repeating the seed changes the partition (that's the whole point of averaging seeds).
"""
import numpy as np

from neuroscan.tasks.workload.feature_importance._cv import Cv


def _data(n_subj=10, per_subj=4, n_feat=3, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n_subj * per_subj, n_feat))
    y = np.tile([0, 1], n_subj * per_subj // 2)
    groups = np.repeat(np.arange(n_subj), per_subj)
    return F, y, groups


def test_grouped_folds_count_is_seeds_times_k():
    F, y, g = _data()
    folds = list(Cv.grouped_folds(F, y, g, seeds=(0, 1, 2), k=5))
    assert len(folds) == 3 * 5                            # one fold per (seed, k) pair


def test_folds_are_subject_disjoint():
    F, y, g = _data()
    for tr, te in Cv.grouped_folds(F, y, g, seeds=(0,), k=5):
        assert set(g[tr]).isdisjoint(set(g[te]))         # no subject on both sides


def test_each_sample_tested_once_per_seed():
    F, y, g = _data()
    tested = np.concatenate([te for _tr, te in Cv.grouped_folds(F, y, g, seeds=(0,), k=5)])
    assert sorted(tested.tolist()) == list(range(len(y)))  # every index in exactly one test fold


def test_reseeding_changes_the_partition():
    F, y, g = _data()
    f0 = [te.tolist() for _tr, te in Cv.grouped_folds(F, y, g, seeds=(0,), k=5)]
    f1 = [te.tolist() for _tr, te in Cv.grouped_folds(F, y, g, seeds=(1,), k=5)]
    assert f0 != f1                                       # different seed -> different split
