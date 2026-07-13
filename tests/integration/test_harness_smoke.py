"""Integration smoke: the harness spine end-to-end on synthetic data, no MOABB/torch needed.

Stubs store.gather so a fold's (X, y) is generated in-memory, feeds a trivial linearly-separable
problem through a real sklearn LDA via the (fit_fn, score_fn) contract, and checks aggregate's shape
+ that a separable signal decodes well. Proves the harness wiring independent of any dataset download.
"""
import numpy as np
import polars as pl

from neuroscan.evaluation import harness


def _fake_fold(name, n_per_class=20, n_ch=4, n_t=8, seed=0):
    # a meta frame whose rows just need 'subject'/'epoch'/'path' columns for gather (which we stub)
    df = pl.DataFrame({"subject": [name] * (4 * n_per_class)})
    return name, df, df


def test_aggregate_shape_and_separable_signal(monkeypatch):
    rng = np.random.default_rng(1)

    def fake_gather(df):
        n = len(df)
        y = np.tile([0, 1, 2, 3], n // 4)[:n]
        # class-mean-shifted gaussians -> linearly separable in the channel-mean feature
        X = rng.normal(size=(n, 4, 8)).astype(np.float32)
        X += (y[:, None, None] * 2.0).astype(np.float32)
        return X, y

    monkeypatch.setattr("core.data.store.gather", fake_gather)

    def fit(X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis()
        clf.fit(X.reshape(len(X), -1), y)
        return clf

    def score(clf, X):
        return clf.predict_proba(X.reshape(len(X), -1))

    folds = [_fake_fold("1", seed=1), _fake_fold("2", seed=2)]
    res = harness.Harness.aggregate(harness.Method("lda_stub", fit, score, 4, "within"), folds)

    assert set(res) >= {"method", "regime", "n_folds", "per_fold", "fold_mean", "pooled", "acc_spread"}
    assert res["n_folds"] == 2
    assert len(res["per_fold"]) == 2
    assert np.array(res["pooled"]["confusion"]).shape == (4, 4)
    assert 0.0 <= res["fold_mean"]["ece"] <= 1.0
    assert res["fold_mean"]["acc"] > 0.8        # separable signal must decode


def test_run_persists_a_model_per_fold(monkeypatch, tmp_path):
    """harness.run must save each fold's trained model under run_dir/models/ (mlflow off)."""
    monkeypatch.setenv("MINDSCAPE_NO_MLFLOW", "1")
    rng = np.random.default_rng(3)

    def fake_gather(df):
        n = len(df)
        y = np.tile([0, 1, 2, 3], n // 4)[:n]
        X = rng.normal(size=(n, 4, 8)).astype(np.float32) + (y[:, None, None] * 2.0).astype(np.float32)
        return X, y

    monkeypatch.setattr("core.data.store.gather", fake_gather)

    def fit(X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis().fit(X.reshape(len(X), -1), y)

    def score(clf, X):
        return clf.predict_proba(X.reshape(len(X), -1))

    folds = [_fake_fold("1", seed=1), _fake_fold("2", seed=2)]
    harness.Harness.run(harness.Method("lda_stub", fit, score, 4, "within"), folds,
                tracking_cfg=harness.TrackConfig(run_dir=tmp_path))

    saved = sorted((tmp_path / "models").glob("*.joblib"))
    assert len(saved) == 2                       # one model per fold
    assert {p.stem for p in saved} == {"model_lda_stub_1", "model_lda_stub_2"}
