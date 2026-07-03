"""fNIRS fixed-recipe study — the shared grouped-fold contract and the recipe registry. The full CV is an
integration concern (needs the dataset); here we pin the pure pieces: folds are subject-disjoint (no leakage)
and every recipe references only real descriptor families."""
import numpy as np

from core.features import family_names
from neuroscan.tasks.workload.feature_importance._cv import grouped_folds
from neuroscan.tasks.workload.feature_importance.recipes import _RECIPES


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
