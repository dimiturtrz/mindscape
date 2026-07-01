"""Published reference ceilings (reference.yaml) — surfaced next to our results so a number is always
read against the literature, never in a vacuum. Cite, don't chase (honesty rule).
"""
from __future__ import annotations

from omegaconf import OmegaConf

from core.config import REPO

_REF = REPO / "reference.yaml"


def ceilings(dataset: str, regime: str) -> dict:
    """{method: {acc, kappa?, source}} for a dataset+regime, or {} if none recorded."""
    if not _REF.exists():
        return {}
    ref = OmegaConf.to_container(OmegaConf.load(_REF), resolve=True)
    return (ref.get(dataset, {}) or {}).get(regime, {}) or {}


def compare(our_acc: float, dataset: str, regime: str, method: str | None = None) -> str:
    """One-line read of our acc vs the recorded ceiling(s). `method` picks one row; else the SOTA row."""
    cs = ceilings(dataset, regime)
    if not cs:
        return f"our acc {our_acc:.3f} (no reference recorded for {dataset}/{regime})"
    key = method if method in cs else ("sota" if "sota" in cs else next(iter(cs)))
    bar = cs[key]
    gap = our_acc - bar["acc"]
    return (f"our acc {our_acc:.3f} vs {key} {bar['acc']:.3f} ({gap:+.3f}) — {bar['source']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print reference ceilings")
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--regime", default="within_subject")
    a = ap.parse_args()
    for m, v in ceilings(a.dataset, a.regime).items():
        k = f" kappa {v['kappa']:.2f}" if "kappa" in v else ""
        print(f"  {m:16} acc {v['acc']:.2f}{k}   [{v['source']}]")
