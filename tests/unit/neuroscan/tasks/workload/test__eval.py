"""Shared fNIRS CV scorer (`_eval.cv_score`) — the within/cross-subject repeated-seeded k-fold loop.

Pure plumbing on synthetic epochs: grouped (StratifiedGroupKFold, subject-disjoint) vs within
(StratifiedKFold), the `build=None -> FnirsLda` default vs a custom decoder thunk, and the `classes=`
two-label restrict + binary relabel. A clean amplitude signal must decode above chance either way.
"""
import numpy as np

from baselines.fnirs.features import FnirsLda
from neuroscan.tasks.workload._eval import CvConfig, CvData, Eval


def _dataset(n_subj=10, per_class=3, n_ch=4, n_t=60, n_classes=2, seed=0):
    """Subject-grouped fNIRS-like epochs `[n, ch, t]` whose class rides in the response AMPLITUDE (each class
    adds a constant offset), so `amplitude_features` (mean/slope/peak) separates them."""
    rng = np.random.default_rng(seed)
    X, y, g = [], [], []
    for s in range(n_subj):
        for cls in range(n_classes):
            for _ in range(per_class):
                X.append(rng.normal(scale=0.5, size=(n_ch, n_t)) + cls * 2.0)
                y.append(cls)
                g.append(s)
    return CvData(np.asarray(X), np.asarray(y), np.asarray(g))


def _valid(acc, sd, kap):
    assert 0.0 <= acc <= 1.0
    assert sd >= 0.0
    assert -1.0 <= kap <= 1.0


def test_grouped_default_build_decodes_amplitude():
    data = _dataset()
    acc, sd, kap = Eval.cv_score(None, data, CvConfig(grouped=True, seeds=(0, 1), k=5))
    _valid(acc, sd, kap)
    assert acc > 0.7                                        # clean amplitude signal -> well above 0.5 chance


def test_within_subject_split_also_decodes():
    data = _dataset()
    acc, sd, kap = Eval.cv_score(None, data, CvConfig(grouped=False, seeds=(0,), k=5))
    _valid(acc, sd, kap)
    assert acc > 0.7


def test_custom_build_thunk_matches_default_fnirslda():
    data = _dataset()
    cfg = CvConfig(grouped=True, seeds=(0,), k=5)
    default = Eval.cv_score(None, data, cfg)
    custom = Eval.cv_score(lambda: FnirsLda(), data, cfg)   # build != None path
    assert np.allclose(default, custom)                    # the thunk builds the same decoder -> same score


def test_classes_filter_restricts_and_binary_relabels():
    """`classes=(0, 2)` on a 3-class set keeps only those labels and relabels binary (2 -> 1). The kept two
    classes are still amplitude-separable, so it decodes; a 3rd untouched class is dropped from scoring."""
    data = _dataset(n_classes=3)
    acc, sd, kap = Eval.cv_score(None, data, CvConfig(grouped=True, seeds=(0,), k=5, classes=(0, 2)))
    _valid(acc, sd, kap)
    assert acc > 0.7                                        # 0 vs 2 have the largest amplitude gap
