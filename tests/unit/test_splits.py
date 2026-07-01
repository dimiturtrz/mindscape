"""Unit tests for split-as-criteria — the honesty spine."""
import polars as pl

from core.data import splits


def _meta():
    # 3 subjects × 2 sessions × 4 epochs
    rows = []
    for s in ("1", "2", "3"):
        for sess in ("0train", "1test"):
            for e in range(4):
                rows.append({"subject": s, "session": sess, "label_id": e % 2, "epoch": e})
    return pl.DataFrame(rows)


def test_test_subjects_holds_out_whole_subject():
    tr, _va, te = splits.make_split(_meta(), test_subjects=["3"])
    assert set(te["subject"].unique()) == {"3"}
    assert "3" not in set(tr["subject"].unique())


def test_test_sessions_holds_out_session_across_subjects():
    tr, _va, te = splits.make_split(_meta(), test_sessions=["1test"])
    assert set(te["session"].unique()) == {"1test"}
    assert set(tr["session"].unique()) == {"0train"}


def test_val_subjects_disjoint_from_train_and_test():
    tr, va, te = splits.make_split(_meta(), test_subjects=["1"], val_subjects=["2"])
    assert set(te["subject"].unique()) == {"1"}
    assert set(va["subject"].unique()) == {"2"}
    assert set(tr["subject"].unique()) == {"3"}


def test_loso_yields_one_fold_per_subject():
    folds = list(splits.leave_one_subject_out(_meta()))
    assert len(folds) == 3
    all_subs = set(_meta()["subject"].unique().to_list())
    for sub, tr, te in folds:
        assert set(te["subject"].unique()) == {sub}
        assert set(tr["subject"].unique()) == all_subs - {sub}     # train = ALL others, in full (no val carve)


def test_within_subject_session_protocol():
    tr, _va, te = splits.within_subject(_meta(), "1", test_sessions=["1test"])
    assert set(tr["subject"].unique()) == {"1"}
    assert set(te["subject"].unique()) == {"1"}
    assert set(te["session"].unique()) == {"1test"}
