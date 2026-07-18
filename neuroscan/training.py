"""Shared training scaffold — the optimization mechanics both trainers hand-rolled (bd 1eca).

`BraindecodeClf` (classification) and `TrainNice` (contrastive retrieval) differ in loss + data + eval, but the
machinery *around* the loss is the same: enable TF32, then run a validation-checkpointed early-stopping loop.
That common part lives here so neither trainer copies it — the losses stay per-trainer, the scaffold is one home.
"""
from __future__ import annotations

from typing import Any

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


class EarlyStopper:
    """Best-checkpoint tracker + early-stop patience, shared by both trainers.

    `mode` is the direction the validation metric improves — "max" for val top-1 (TrainNice), "min" for val loss
    (BraindecodeClf); `min_delta` is the improvement margin. `update` records a CPU-cloned best `state_dict` when
    the metric improves and returns True once `patience` epochs pass with no improvement (the caller breaks);
    `restore` loads the best checkpoint back into the module. Internally the metric is stored sign-normalized so a
    single `>` comparison serves both directions.
    """

    def __init__(self, patience: int, *, mode: str = "max", min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.sign = 1.0 if mode == "max" else -1.0
        self._best = -float("inf")       # sign-normalized: always "higher is better" internally
        self.best_state: dict[str, Any] | None = None
        self.best_step = -1
        self.bad = 0

    def update(self, metric: float, module: torch.nn.Module, step: int = -1) -> bool:
        """Record the checkpoint if `metric` improved past `min_delta`; return True when patience is exhausted."""
        if self.sign * metric > self._best + self.min_delta:
            self._best, self.best_step, self.bad = self.sign * metric, step, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        else:
            self.bad += 1
        return self.patience > 0 and self.bad >= self.patience

    @property
    def best_metric(self) -> float:
        """The best validation metric seen, in the caller's original orientation."""
        return self.sign * self._best

    def restore(self, module: torch.nn.Module) -> None:
        """Load the best-checkpoint weights back (no-op if the metric never improved, e.g. a no-val run)."""
        if self.best_state is not None:
            module.load_state_dict(self.best_state)
