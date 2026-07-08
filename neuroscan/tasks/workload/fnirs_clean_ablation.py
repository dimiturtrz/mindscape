"""Does physiological-noise cleaning help fNIRS? — ablate the preprocessing Cleaner stage under a FIXED
decoder. The decoder is held constant (the best fair fNIRS method, mean+slope+peak -> shrinkage-LDA), so any
within/cross delta is the CLEANING, nothing else. Each arm re-epochs the data with a different `clean` spec
(None | cbsi | detrend | composite) and scores within-subject and cross-subject.

    python -m neuroscan.tasks.workload.fnirs_clean_ablation
"""
from __future__ import annotations

from core.data import store
from core.data.fnirs.base import FnirsCfg
from neuroscan.tasks.workload._eval import cv_score

_DATASET = "shin2017_nback"
# clean specs to compare — None is the current baseline; cbsi is the flagship; detrend a near-null control;
# the pair shows the composite chain works.
_ARMS = [("none (baseline)", None), ("cbsi", "cbsi"), ("detrend", "detrend"), ("cbsi+detrend", ["cbsi", "detrend"])]


def main():
    print("fNIRS cleaner ablation · Shin n-back · fixed decoder fnirs_lda · 3x5-fold · chance 0.333\n")
    print(f"  {'clean':<18}{'within':>9}{'±sd':>7}{'κ':>7}   {'cross':>9}{'±sd':>7}{'κ':>7}{'  Δcross':>9}")
    base_cross = None
    for name, spec in _ARMS:
        meta = store.load(_DATASET, FnirsCfg(clean=spec))
        X, y = store.gather(meta)
        groups = meta["subject"].to_numpy()
        wa, ws, wk = cv_score(None, X, y, groups, grouped=False)
        ca, cs, ck = cv_score(None, X, y, groups, grouped=True)
        if base_cross is None:
            base_cross = ca
        dc = "" if spec is None else f"{ca - base_cross:+.3f}"
        print(f"  {name:<18}{wa:>9.3f}{ws:>7.3f}{wk:>7.3f}   {ca:>9.3f}{cs:>7.3f}{ck:>7.3f}{dc:>9}")
    print("\n  Δcross vs no-clean baseline. >0 = cleaning helps cross-subject; ≈0 = the noise it removes wasn't\n"
          "  limiting here (expected if the ceiling is construct, not noise — cleaning still aids fusion hygiene).")


if __name__ == "__main__":
    main()
