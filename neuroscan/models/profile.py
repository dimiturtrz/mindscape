"""Model size profiling — parameters + FLOPs per decoder, for the honest deployability table.

FLOPs are a single forward pass at the real 2a input (22 ch x 1125 samples, batch 1), via fvcore.
Params are the trainable count. The point: a tiny edge-deployable net vs a heavier near-SOTA one,
measured rather than asserted.

    python -m neuroscan.models.profile
"""
from __future__ import annotations

N_CHANS, N_TIMES, N_CLASSES = 22, 1125, 4


def profile(cls: str, n_chans=N_CHANS, n_times=N_TIMES, n_classes=N_CLASSES) -> dict:
    import braindecode.models as M
    import torch

    net = getattr(M, cls)(n_chans=n_chans, n_outputs=n_classes, n_times=n_times).eval()
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    dummy = torch.zeros(1, n_chans, n_times)
    flops = None
    try:
        import logging

        from fvcore.nn import FlopCountAnalysis
        logging.getLogger("fvcore").setLevel(logging.ERROR)
        flops = int(FlopCountAnalysis(net, dummy).unsupported_ops_warnings(False)
                    .uncalled_modules_warnings(False).total())
    except Exception as e:
        print(f"  ({cls}: FLOPs unavailable: {e})")
    return {"model": cls, "params": int(params), "flops": flops}


def _fmt(n):
    if n is None:
        return "—"
    for unit, div in (("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n/div:.2f}{unit}"
    return str(n)


def main():
    from neuroscan.models.decoders import MODELS
    rows = [profile(cfg["cls"]) for cfg in MODELS.values()]
    print(f"\n=== params + FLOPs (input {N_CHANS}ch x {N_TIMES} samples, batch 1) ===")
    print(f"{'model':16} {'params':>10} {'FLOPs':>10}")
    for r in rows:
        print(f"{r['model']:16} {_fmt(r['params']):>10} {_fmt(r['flops']):>10}")
    return rows


if __name__ == "__main__":
    main()
