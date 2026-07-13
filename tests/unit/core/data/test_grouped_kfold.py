"""The grouped k-fold (BenchNIRS 'generalised') cross-subject regime — each subject tested exactly once,
train/test subject-disjoint (no leakage)."""
import polars as pl

from core.data import splits


def _meta(n_subjects=26, per=3):
    return pl.DataFrame({"subject": [str(s) for s in range(n_subjects) for _ in range(per)],
                         "session": ["0"] * (n_subjects * per), "run": ["0"] * (n_subjects * per)})


def test_grouped_kfold_partitions_subjects_once():
    folds = list(splits.Splits.grouped_kfold(_meta(26), k=5))
    assert len(folds) == 5
    test_subs = [set(te["subject"].unique().to_list()) for _, _, te in folds]
    union = set().union(*test_subs)
    assert union == {str(s) for s in range(26)}                     # every subject tested
    assert sum(len(s) for s in test_subs) == 26                     # exactly once (no overlap)


def test_grouped_kfold_train_test_disjoint_and_full():
    for _, tr, te in splits.Splits.grouped_kfold(_meta(26), k=5):
        tr_s = set(tr["subject"].unique().to_list())
        te_s = set(te["subject"].unique().to_list())
        assert tr_s.isdisjoint(te_s)                                # no subject leaks across the split
        assert tr_s | te_s == {str(s) for s in range(26)}           # train = ALL non-test (full, no val carve)
