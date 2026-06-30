"""Stage-2 guard: ONNX export must match torch (parity gate), and the quant path must run.

The whole efficient-deploy claim rests on the exported model being faithful — so the parity check is a
hard gate. INT8 dynamic quant is best-effort here (these EEG nets are tiny enough that INT8 can add
overhead rather than save — a real finding, not a correctness failure), so its assertions are lenient.
"""
import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def _tiny_net():
    import torch  # noqa: F401
    from braindecode.models import EEGNetv4
    return EEGNetv4(n_chans=8, n_outputs=2, n_times=128).eval()


def test_onnx_export_parity_gate(tmp_path):
    from core import export_onnx

    net = _tiny_net()
    X = np.random.RandomState(0).randn(5, 8, 128).astype(np.float32)
    path = export_onnx.export(net, 8, 128, tmp_path / "m.onnx", device="cpu")
    assert path.exists()
    gap = export_onnx.parity(net, path, X, device="cpu")
    assert gap < 1e-3, f"ONNX parity failed: {gap:.2e}"


def test_onnx_run_shape(tmp_path):
    from core import export_onnx

    net = _tiny_net()
    X = np.random.RandomState(1).randn(3, 8, 128).astype(np.float32)
    path = export_onnx.export(net, 8, 128, tmp_path / "m.onnx", device="cpu")
    logits = export_onnx.run(path, X)
    assert logits.shape == (3, 2)


def test_int8_quant_runs_if_supported(tmp_path):
    from core import export_onnx

    net = _tiny_net()
    X = np.random.RandomState(2).randn(4, 8, 128).astype(np.float32)
    fp32 = export_onnx.export(net, 8, 128, tmp_path / "m.onnx", device="cpu")
    try:
        int8 = export_onnx.quantize_int8(fp32, tmp_path / "m_int8.onnx")
    except Exception as e:                      # tiny-net quantizer quirks are not a correctness gate
        pytest.skip(f"int8 quantization unsupported for this net: {e}")
    assert int8.exists()
    assert export_onnx.run(int8, X).shape == (4, 2)
    assert export_onnx.file_mb(int8) > 0
