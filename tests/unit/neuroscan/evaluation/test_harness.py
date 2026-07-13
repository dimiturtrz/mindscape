"""harness — the eval spine: `folds_for` (regime -> fold list) and `aggregate` (pure fold -> metrics).

`store.gather` (the disk read) is stubbed so only the aggregation LOGIC is under test: each synthetic fold
yields known (X, y); a trivial perfect scorer makes acc/kappa exact and the confusion diagonal. `folds_for`
runs on a small synthetic meta cloud (within session-holdout, LOSO, cross_subject_kfold, unknown-regime raise).
"""
import numpy as np
import polars as pl
import pytest

from neuroscan.evaluation import harness


# --- folds_for: turning the data cloud into evaluation-regime folds -------------------------------------------

def _meta():
    rows = []
    for s in ("1", "2", "3"):
        for sess in ("0train", "1test"):
            for e in range(3):
                rows.append({"subject": s, "session": sess, "label_id": e % 2, "epoch": e})
    return pl.DataFrame(rows)


def test_within_folds_one_per_subject_with_session_holdout():
    folds = harness.Harness.folds_for(_meta(), "within", test_sessions=["1test"])
    assert len(folds) == 3
    for name, train, test in folds:
        assert set(train["subject"].unique()) == {name}
        assert set(test["subject"].unique()) == {name}
        assert set(test["session"].unique()) == {"1test"}
        assert set(train["session"].unique()) == {"0train"}


def test_cross_subject_folds_are_loso():
    folds = harness.Harness.folds_for(_meta(), "cross_subject")
    assert len(folds) == 3
    for name, train, test in folds:
        assert set(test["subject"].unique()) == {name}
        assert name not in set(train["subject"].unique())


def test_unknown_regime_raises():
    with pytest.raises(ValueError):
        harness.Harness.folds_for(_meta(), "nope")


def _kfold_meta(n_subjects=5):
    rows = [{"subject": str(s), "session": "0", "label_id": e % 2, "epoch": e}
            for s in range(n_subjects) for e in range(4)]
    return pl.DataFrame(rows)


def test_folds_for_cross_subject_kfold_partitions_subjects():
    folds = harness.Harness.folds_for(_kfold_meta(5), "cross_subject_kfold")
    assert len(folds) == 5                        # k=5 over 5 subjects == LOSO limit
    tested = set()
    for _name, train, test in folds:
        te_subs = set(test["subject"].unique())
        assert te_subs.isdisjoint(set(train["subject"].unique()))
        tested |= te_subs
    assert tested == {str(s) for s in range(5)}   # every subject tested exactly once


# --- aggregate: the pure fold -> metrics spine (fold-mean + pooled) -------------------------------------------

def _fold(name, labels):
    df = pl.DataFrame({"label_id": list(labels)})
    return (name, df, df)                       # train == test frame; gather is stubbed anyway


def _stub_gather(df):
    y = df["label_id"].to_numpy()
    X = y.astype(np.float32).reshape(-1, 1, 1)  # shape only matters as a carrier
    return X, y


def _perfect_method(n_classes):
    # score = one-hot on the true class (X carries y in its single feature) -> acc/kappa == 1
    def fit(_X, _y):
        return object()

    def score(_clf, X):
        y = X.reshape(-1).astype(int)
        return np.eye(n_classes)[y]

    return harness.Method(name="perfect", fit=fit, score=score, n_classes=n_classes, regime="within")


def test_aggregate_fold_mean_and_pooled_perfect_scorer(monkeypatch):
    monkeypatch.setattr(harness.store.Store, "gather", staticmethod(_stub_gather))
    folds = [_fold("s1", [0, 1, 2, 3]), _fold("s2", [0, 0, 1, 1, 2])]   # unequal fold sizes
    res = harness.Harness.aggregate(_perfect_method(4), folds)

    assert res["method"] == "perfect" and res["n_folds"] == 2
    assert res["fold_mean"]["acc"] == 1.0 and res["fold_mean"]["kappa"] == 1.0
    assert res["pooled"]["acc"] == 1.0
    assert [r["fold"] for r in res["per_fold"]] == ["s1", "s2"]
    assert res["per_fold"][0]["n"] == 4 and res["per_fold"][1]["n"] == 5
    conf = np.array(res["pooled"]["confusion"])
    assert conf.shape == (4, 4)
    assert conf.trace() == 9                     # all 9 epochs on the diagonal (perfect)
    assert res["acc_spread"]["min"] == 1.0 and res["acc_spread"]["max"] == 1.0


def test_aggregate_collects_models_out_in_fold_order(monkeypatch):
    monkeypatch.setattr(harness.store.Store, "gather", staticmethod(_stub_gather))
    folds = [_fold("a", [0, 1]), _fold("b", [1, 0])]
    models: list = []
    harness.Harness.aggregate(_perfect_method(2), folds, models_out=models)
    assert [name for name, _clf in models] == ["a", "b"]
