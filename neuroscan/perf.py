"""Device-level training performance switches shared by every training path (bd 1eca, bd 62ak).

The one knob that is parity-safe and worth flipping on the 5090: TF32 for the residual fp32 matmuls. Both
trainers (`BraindecodeClf`, `TrainNice`) call it before building their optimizer.
"""
from __future__ import annotations

import torch


class TorchPerf:
    """Device-level performance switches shared by every training path (op-namespace of staticmethods)."""

    @staticmethod
    def enable_fast_matmul(device: str) -> None:
        """TF32 for the residual fp32 matmuls (`high` precision). Measured −22% step time (bd 62ak: the win is in
        backward, 25.8→17.3 ms) — parity-safe since both loops already run bf16 autocast and TF32's 10-bit mantissa
        is MORE precise than the bf16 already in use. cudnn.benchmark is deliberately NOT set: measured neutral/worse
        for the small convs, and variable end-of-epoch batch shapes would re-trigger its autotune."""
        if device != "cuda":
            return
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
