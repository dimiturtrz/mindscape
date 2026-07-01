"""harness.folds_for — turning the data cloud into evaluation-regime folds."""
import polars as pl

from neuroscan.evaluation import harness


def _meta():
    rows = []
    for s in ("1", "2", "3"):
        for sess in ("0train", "1test"):
            for e in range(3):
                rows.append({"subject": s, "session": sess, "label_id": e % 2, "epoch": e})
    return pl.DataFrame(rows)


def test_within_folds_one_per_subject_with_session_holdout():
    folds = harness.folds_for(_meta(), "within", test_sessions=["1test"])
    assert len(folds) == 3
    for name, train, test in folds:
        assert set(train["subject"].unique()) == {name}
        assert set(test["subject"].unique()) == {name}
        assert set(test["session"].unique()) == {"1test"}
        assert set(train["session"].unique()) == {"0train"}


def test_cross_subject_folds_are_loso():
    folds = harness.folds_for(_meta(), "cross_subject")
    assert len(folds) == 3
    for name, train, test in folds:
        assert set(test["subject"].unique()) == {name}
        assert name not in set(train["subject"].unique())


def test_unknown_regime_raises():
    import pytest
    with pytest.raises(ValueError):
        harness.folds_for(_meta(), "nope")
