"""fNIRS fixed-recipe study — the shared grouped-fold contract and the recipe registry. The full CV is an
integration concern (needs the dataset); here we pin the pure pieces: folds are subject-disjoint (no leakage)
and every recipe references only real descriptor families."""
import numpy as np

from core.features import extract_bank, family_names
from neuroscan.tasks.workload.feature_importance._cv import grouped_folds
from neuroscan.tasks.workload.feature_importance.recipes import _RECIPES, _cv


def test_recipes_reference_only_real_families():
    known = set(family_names())
    for key, (_label, fams) in _RECIPES.items():
        assert fams, f"{key} is empty"
        assert set(fams) <= known, f"{key} references unknown families {set(fams) - known}"


def test_grouped_folds_are_subject_disjoint():
    rng = np.random.default_rng(0)
    n_subj, per = 8, 9
    y = np.array([b % 3 for _ in range(n_subj) for b in range(per)])
    groups = np.array([s for s in range(n_subj) for _ in range(per)])
    F = rng.standard_normal((len(y), 5))
    n_folds = 0
    for tr, te in grouped_folds(F, y, groups, seeds=[0, 1], k=3):
        assert set(groups[tr]).isdisjoint(set(groups[te]))       # whole subjects per side — no leakage
        n_folds += 1
    assert n_folds == 2 * 3                                       # seeds x k


def test_cv_selects_families_and_scores_above_chance():
    """`_cv` restricts the bank to the requested families, runs the shared grouped folds, and returns
    (acc, sd, kappa). A class-separable amplitude signal decodes above 2-class chance."""
    rng = np.random.default_rng(0)
    n_subj, per = 8, 4
    X, y, groups = [], [], []
    for s in range(n_subj):
        for cls in (0, 1):
            for _ in range(per):
                X.append(rng.normal(scale=0.4, size=(6, 40)) + cls * 2.0)   # class rides in the mean/amplitude
                y.append(cls)
                groups.append(s)
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    F, fam = extract_bank(X)
    acc, sd, kap = _cv(F, fam, y, groups, ["mean", "slope"])
    assert 0.0 <= acc <= 1.0 and sd >= 0.0 and -1.0 <= kap <= 1.0
    assert acc > 0.7
    # the family filter really subsets columns: 'mean' alone uses fewer columns than 'mean'+'slope'
    assert (fam == "mean").sum() < np.isin(fam, ["mean", "slope"]).sum()
