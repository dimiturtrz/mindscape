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


def test_resolve_registers_every_models_class():
    """Every `cls` string in MODELS resolves through the explicit registry to an nn.Module subclass —
    no MODELS entry points at an unregistered braindecode class."""
    import torch

    from neuroscan.models import decoders

    for spec in decoders.MODELS.values():
        cls = decoders.BraindecodeClf.resolve(spec["cls"])
        assert issubclass(cls, torch.nn.Module)


def test_resolve_unknown_class_raises_listing_registry():
    from neuroscan.models import decoders

    with pytest.raises(KeyError, match="unregistered braindecode class"):
        decoders.BraindecodeClf.resolve("NotANet")
