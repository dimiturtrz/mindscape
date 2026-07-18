"""Model size profiling — parameters + FLOPs per decoder, for the measured deployability table.

FLOPs are a single forward pass at the real 2a input (22 ch x 1125 samples, batch 1), via fvcore.
Params are the trainable count. The point: a tiny edge-deployable net vs a heavier near-SOTA one,
measured rather than asserted.

    python -m neuroscan.models.profile
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from neuroscan.models.decoders import MODELS
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)

N_CHANS, N_TIMES, N_CLASSES = 22, 1125, 4


class Profile:
    @classmethod
    def profile(cls, model_cls: type[torch.nn.Module], n_chans: int = N_CHANS, n_times: int = N_TIMES,
                n_classes: int = N_CLASSES) -> dict[str, Any]:
        net = model_cls(n_chans=n_chans, n_outputs=n_classes, n_times=n_times).eval()
        params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        dummy = torch.zeros(1, n_chans, n_times)
        flops = None
        try:
            from fvcore.nn import FlopCountAnalysis  # noqa: PLC0415
            logging.getLogger("fvcore").setLevel(logging.ERROR)
            flops = int(FlopCountAnalysis(net, dummy).unsupported_ops_warnings(enabled=False)
                        .uncalled_modules_warnings(enabled=False).total())
        except Exception as e:  # noqa: BLE001
            logger.info(f"  ({model_cls.__name__}: FLOPs unavailable: {e})")
        return {"model": model_cls.__name__, "params": int(params), "flops": flops}

    @classmethod
    def _fmt(cls, n: int | None) -> str:
        if n is None:
            return "—"
        for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
            if n >= div:
                return f"{n/div:.2f}{unit}"
        return str(n)


    @classmethod
    def main(cls):
        Cli.setup_logging()
        rows = [cls.profile(cfg["cls"]) for cfg in MODELS.values()]
        logger.info(f"\n=== params + FLOPs (input {N_CHANS}ch x {N_TIMES} samples, batch 1) ===")
        logger.info(f"{'model':16} {'params':>10} {'FLOPs':>10}")
        for r in rows:
            logger.info(f"{r['model']:16} {cls._fmt(r['params']):>10} {cls._fmt(r['flops']):>10}")
        return rows


if __name__ == "__main__":
    Profile.main()
