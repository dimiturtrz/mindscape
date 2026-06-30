"""Integration: train a real decoder -> ONNX export -> parity, UNMOCKED (CPU, tiny).

Exercises the decoders -> export_onnx chain on a real braindecode net trained for 2 epochs, then checks
the exported model matches torch (the gate the whole Stage-2 deploy claim rests on)."""
import numpy as np
import pytest

pytest.importorskip("braindecode")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def test_train_export_parity_chain(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from core import export_onnx
    from neuroscan.models import decoders

    rng = np.random.default_rng(0)
    n, ch, t = 24, 8, 128
    X = (rng.normal(size=(n, ch, t)) + np.tile([0, 1, 2, 3], n // 4)[:, None, None]).astype(np.float32)
    y = np.tile([0, 1, 2, 3], n // 4)

    fit, score = decoders.make("eegnet")
    clf = fit(X, y, epochs=2, patience=0, crop_frac=None, standardize="zscore")
    probs = score(clf, X)
    assert probs.shape == (n, 4)
    assert clf.device == "cpu"

    path = export_onnx.export(clf.net, ch, t, tmp_path / "m.onnx", device="cpu")
    gap = export_onnx.parity(clf.net, path, clf.std(X), device="cpu")
    assert gap < 1e-3, f"export parity failed: {gap:.2e}"
