"""CPU smoke for the crop-augmenting trainer — contract + shapes, no GPU/download.

Forces CPU (monkeypatch cuda off) so it never contends with a real GPU run, and runs a 2-epoch fit on
tiny synthetic trials. Checks: crop_len derives from crop_frac, per-trial proba shape, rows sum to 1.
"""
import numpy as np
import pytest

pytest.importorskip("braindecode")


def test_crop_trainer_cpu_smoke(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from neuroscan.models import decoders

    rng = np.random.default_rng(0)
    n, ch, T = 24, 8, 256
    X = rng.standard_normal((n, ch, T)).astype(np.float32)
    y = np.tile([0, 1, 2, 3], n // 4)

    fit, score = decoders.BraindecodeClf.make("eegnet")
    clf = fit(X, y, epochs=2, patience=0, log_every=0)
    assert clf.crop_len == int(0.5 * T)          # crop_frac default 0.5
    assert clf.device == "cpu"

    p = score(clf, X)
    assert p.shape == (n, 4)                      # per-trial, crops averaged back
    assert np.allclose(p.sum(1), 1.0, atol=1e-4)


def test_every_models_cls_is_an_nn_module():
    """Each MODELS entry's `cls` is a braindecode nn.Module class (not a name to look up) — a non-net entry
    would fail here rather than at fit time."""
    import torch

    from neuroscan.models import decoders

    for spec in decoders.MODELS.values():
        assert issubclass(spec["cls"], torch.nn.Module)


def test_make_unknown_method_raises_listing_methods():
    from neuroscan.models import decoders

    with pytest.raises(KeyError, match="unknown decoder"):
        decoders.BraindecodeClf.make("not_a_method")


def _tiny_xy(seed=0):
    rng = np.random.default_rng(seed)
    n, ch, T = 24, 8, 256
    return rng.standard_normal((n, ch, T)).astype(np.float32), np.tile([0, 1, 2, 3], n // 4)


def test_estimator_matches_make_route_byte_identical(monkeypatch):
    """The sklearn-estimator face (bd kvb) delegates to `make`, so its per-trial proba is IDENTICAL to the
    (fit, score) harness route at the same seed/overrides — additive interoperability, not a second trainer."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from neuroscan.models import decoders

    X, y = _tiny_xy()
    over = {"epochs": 2, "patience": 0}

    fit, score = decoders.BraindecodeClf.make("eegnet")
    p_route = score(fit(X, y, **over), X)

    est = decoders.BraindecodeEstimator("eegnet", **over).fit(X, y)
    p_est = est.predict_proba(X)

    assert np.array_equal(p_est, p_route)          # same construction path + seed -> byte-identical


def test_estimator_composes_in_sklearn_pipeline(monkeypatch):
    """It drops into a sklearn Pipeline unchanged (the kvb acceptance) and predict() returns class labels."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from sklearn.pipeline import Pipeline

    from neuroscan.models import decoders

    X, y = _tiny_xy()
    pipe = Pipeline([("clf", decoders.BraindecodeEstimator("eegnet", epochs=2, patience=0))])
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)
    assert set(np.unique(pred)).issubset(set(np.unique(y)))     # predictions are real class labels


def test_estimator_get_set_params_and_clone():
    """sklearn contract: hyperparameters round-trip through get_params/set_params and survive clone (no fit,
    no torch, no GPU) — what GridSearchCV relies on."""
    from sklearn.base import clone

    from neuroscan.models import decoders

    est = decoders.BraindecodeEstimator("eegnet", lr=1e-3)
    assert est.get_params()["lr"] == 1e-3
    est.set_params(lr=5e-4, method="atcnet")
    assert est.get_params()["lr"] == 5e-4
    assert clone(est).get_params()["method"] == "atcnet"        # clone reconstructs from get_params
