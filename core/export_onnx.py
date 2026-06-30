"""ONNX export + INT8 quantization + parity/size/latency — the Stage-2 edge tail of training.

The deployable artifact is the trained decoder, exported to ONNX and dynamically quantized to INT8 for
commodity edge CPU. We report the honest deployment triad (the siblings' discipline, carried):
    accuracy:  fp32 vs int8           (the cost of quantizing)
    size:      fp32 vs int8 (MB)      (the compression)
    latency:   fp32 vs int8 (ms, CPU) (the speedup on the edge target)

Parity-gated: the fp32 ONNX must match the torch model before we trust the quantized one. Deploy
standardizes input the same way training did (per-channel z-score) — the net itself is exported raw.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def export(net, n_chans: int, n_times: int, path: str | Path, device: str = "cpu") -> Path:
    """Export a trained torch module to ONNX with a dynamic batch axis. Returns the path."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    net = net.to(device).eval()
    dummy = torch.zeros(1, n_chans, n_times, device=device)
    torch.onnx.export(
        net, dummy, str(path), input_names=["eeg"], output_names=["logits"],
        dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    return path


def _session(path: str | Path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1            # single-thread = realistic edge latency
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(path: str | Path, X: np.ndarray) -> np.ndarray:
    """ONNX logits for standardized input X [n, ch, t]."""
    sess = _session(path)
    return sess.run(["logits"], {"eeg": X.astype(np.float32)})[0]


def parity(net, onnx_path: str | Path, X_std: np.ndarray, device: str = "cpu") -> float:
    """Max abs difference between torch logits and ONNX logits on the same input. Gate before trusting."""
    import torch

    net = net.to(device).eval()
    with torch.no_grad():
        tlog = net(torch.tensor(X_std.astype(np.float32), device=device)).cpu().numpy()
    return float(np.abs(tlog - run(onnx_path, X_std)).max())


def quantize_int8(fp32_path: str | Path, int8_path: str | Path) -> Path:
    """Dynamic weight-only INT8 quantization (no calibration set needed). Returns the int8 path."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path = Path(int8_path)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    return int8_path


def file_mb(path: str | Path) -> float:
    return round(Path(path).stat().st_size / 1e6, 3)


def latency_ms(path: str | Path, X_std: np.ndarray, runs: int = 100, warmup: int = 10) -> float:
    """Mean single-sample inference latency (ms) on ORT CPU — the edge-realistic number."""
    sess = _session(path)
    x = X_std[:1].astype(np.float32)
    for _ in range(warmup):
        sess.run(["logits"], {"eeg": x})
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(["logits"], {"eeg": x})
    return round((time.perf_counter() - t0) / runs * 1e3, 3)
