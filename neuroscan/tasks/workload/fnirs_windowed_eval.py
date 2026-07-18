"""Does keeping the fNIRS time axis help? — the collapse baseline vs the windowed stage-1 decoder, under a
matched protocol, within AND cross-subject. The ONLY difference between the two arms is the representation
(one global mean/slope/peak per channel vs per-sub-window descriptors soft-voted to the block), so a delta
isolates "does not-collapsing time help", nothing else.

Two regimes, because they answer different questions:
  - within-subject  (StratifiedKFold, subjects in train AND test): does the richer representation carry more
    signal at all? the optimistic number.
  - cross-subject   (StratifiedGroupKFold by subject, unseen subjects): the number the literature reports a
    ~0.39 ceiling for — measured only on the collapsed representation, untested for windowed on Shin.

    python -m neuroscan.tasks.workload.fnirs_windowed_eval
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

from baselines.fnirs.windowed import WindowedConfig, WindowedFnirs
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload._eval import CvConfig, CvData, Eval

logger = logging.getLogger(__name__)

_FS = 10.0
_DATASET = "shin2017_nback"

# the aggregation IS the design axis — sweep it, don't bet on one. Plus a window granularity (coarse vs fine):
# coarse keeps the pooled stage-1 low-dim / concat tractable; fine gives temporal resolution but more to overfit.
# concat carried the only within-subject gain; trace it across a granularity ladder (very-coarse -> fine) to
# see if fewer windows (less to overfit) let the gain survive transfer. Keep one pooled tier as the ruled-out
# control (mean/max/lse were all a wash below collapse). One assignment (no import-time side effect).
_ARMS: list[tuple[str, Callable[[], Any] | None]] = [
    ("collapse (baseline)", None),
    *((f"windowed concat · {_label}",
       lambda win_s=_win_s, hop_s=_hop_s:
           WindowedFnirs(WindowedConfig(win_s=win_s, hop_s=hop_s, fs=_FS, aggregate="concat")))
      for _win_s, _hop_s, _label in [(11.0, 11.0, "vcoarse 2w"), (7.0, 7.0, "coarse 3w"),
                                     (6.0, 3.0, "med 6/3"), (4.0, 1.0, "fine 4/1")]),
    *((f"windowed {_agg} · med 6/3",
       lambda agg=_agg: WindowedFnirs(WindowedConfig(win_s=6.0, hop_s=3.0, fs=_FS, aggregate=agg)))
      for _agg in ("mean", "max", "lse")),
]


def main():
    Cli.setup_logging()
    meta = store.Store.load(_DATASET, cast(EpochCfg, FnirsCfg()))
    X, y = store.Store.gather(meta)
    groups = meta["subject"].to_numpy()
    chance = 1.0 / (int(y.max()) + 1)
    logger.info(f"fNIRS windowed-aggregation sweep vs collapse · Shin n-back · {len(y)} blocks · "
          f"{meta['subject'].n_unique()} subj · 3x5-fold · chance {chance:.3f}\n")
    logger.info(f"  {'arm':<26}{'within':>9}{'±sd':>7}   {'cross':>9}{'±sd':>7}{'  Δcross':>9}")
    data = CvData(X, y, groups)
    base_cross = None
    for name, build in _ARMS:
        wa, ws, _ = Eval.cv_score(build, data, CvConfig(grouped=False))
        ca, cs, _ = Eval.cv_score(build, data, CvConfig(grouped=True))
        if base_cross is None:
            base_cross = ca
        dc = "" if build is None else f"{ca - base_cross:+.3f}"
        logger.info(f"  {name:<26}{wa:>9.3f}{ws:>7.3f}   {ca:>9.3f}{cs:>7.3f}{dc:>9}")
    logger.info("\n  Δcross = cross-subject acc minus the collapse baseline. >0 for ANY aggregation = keeping time\n"
          "  survives transfer; all ≤0 = the collapse is genuinely adequate for this signal (then: DA, not features).")


if __name__ == "__main__":
    main()
