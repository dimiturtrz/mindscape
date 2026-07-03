"""fNIRS Optuna study — the pure pieces: the CV objective returns a sane accuracy, and the stability
metric reads top-family agreement across seeds correctly (the deliverable's validity check)."""
import numpy as np

from core.features import extract_bank, family_names
from neuroscan.tasks.workload.optuna_fnirs import _cv_score, _stability


def test_cv_score_returns_accuracy_in_range():
    rng = np.random.default_rng(0)
    # 6 subjects x 12 blocks, class encoded in a mean offset so a decoder beats chance
    X, y, groups = [], [], []
    for s in range(6):
        for b in range(12):
            c = b % 3
            X.append(rng.standard_normal((8, 40)) + c)          # [ch, t] shifted by class
            y.append(c); groups.append(s)
    X = np.asarray(X); y = np.asarray(y); groups = np.asarray(groups)
    F, fam = extract_bank(X)
    acc = _cv_score(F, fam, y, groups, {f: 1.0 for f in family_names()}, fold_seeds=[0], k=3)
    assert 0.0 <= acc <= 1.0
    assert acc > 1 / 3                                           # separable signal -> beats 3-class chance


def test_stability_high_when_top_families_agree():
    fams = [f"f{i}" for i in range(10)]
    same = {f"f{i}": (10 - i) for i in range(10)}               # identical ranking across "seeds"
    st = _stability([same, dict(same), dict(same)], fams, topn=5)
    assert st["mean_jaccard"] == 1.0                            # perfect agreement
    assert st["consensus_order"][0] == "f0"


def test_stability_low_when_top_families_disagree():
    fams = [f"f{i}" for i in range(10)]
    a = {f"f{i}": (10 - i) for i in range(10)}                  # top = f0..f4
    b = {f"f{i}": i for i in range(10)}                         # top = f9..f5 (disjoint)
    st = _stability([a, b], fams, topn=5)
    assert st["mean_jaccard"] == 0.0                            # no overlap -> unstable
