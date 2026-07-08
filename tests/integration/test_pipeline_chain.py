"""Integration: the full data cloud -> splits -> harness chain on synthetic npz, UNMOCKED.

Writes real per-subject npz + a meta frame shaped like store.load produces, then runs the whole
store.gather -> splits -> folds_for -> aggregate -> metrics chain. Exercises the modules together
(the unit smoke monkeypatches store.gather; this one doesn't)."""
import numpy as np
import polars as pl

from neuroscan.evaluation import harness


def _write_cloud(tmp_path, n_sub=3, per_class=12):
    rows = []
    for s in range(1, n_sub + 1):
        rng = np.random.default_rng(s)
        n = per_class * 4
        y = np.tile([0, 1, 2, 3], per_class)
        X = (rng.normal(size=(n, 4, 16)) + y[:, None, None] * 2.0).astype(np.float32)  # separable
        half = n // 2
        sess = np.array(["0train"] * half + ["1test"] * (n - half))
        npz = tmp_path / f"sub{s}.npz"
        np.savez(npz, X=X, y=y, session=sess, run=np.array(["0"] * n))
        for i in range(n):
            rows.append({"dataset": "syn", "subject": str(s), "session": str(sess[i]), "run": "0",
                         "label_id": int(y[i]), "label": str(int(y[i])), "epoch": i,
                         "file": npz.name, "path": str(npz)})
    return pl.DataFrame(rows)


def _lda():
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    def fit(X, y):
        c = LinearDiscriminantAnalysis()
        c.fit(X.reshape(len(X), -1), y)
        return c

    def score(c, X):
        return c.predict_proba(X.reshape(len(X), -1))
    return fit, score


def test_within_chain_decodes_separable(tmp_path):
    meta = _write_cloud(tmp_path)
    folds = harness.folds_for(meta, "within", test_sessions=["1test"])
    assert len(folds) == 3
    fit, score = _lda()
    res = harness.aggregate(harness.Method("lda", fit, score, 4, "within"), folds)
    assert res["fold_mean"]["acc"] > 0.8                  # separable signal decodes through the chain
    assert np.array(res["pooled"]["confusion"]).shape == (4, 4)


def test_cross_subject_chain_runs(tmp_path):
    meta = _write_cloud(tmp_path)
    folds = harness.folds_for(meta, "cross_subject")
    fit, score = _lda()
    res = harness.aggregate(harness.Method("lda", fit, score, 4, "cross_subject"), folds)
    assert res["n_folds"] == 3                            # leave-one-subject-out -> one fold per subject
    assert 0.0 <= res["fold_mean"]["ece"] <= 1.0
