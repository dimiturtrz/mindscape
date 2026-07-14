"""GLM-β vs the mean/slope/peak collapse — does a model-based (HRF-template) amplitude beat the hand-collapsed
features on fNIRS n-back? Fixed everything else (shrinkage-LDA, subject-grouped CV, no cleaning), so the delta
is the feature. Within AND cross-subject; plus the per-boundary binary breakdown, because the interesting
question is whether GLM-β sharpens the *decodable* boundary (0-vs-load) or — the real test — finally cracks
2-vs-3 (which the collapse and the literature say is physiologically absent).

    python -m neuroscan.tasks.workload.fnirs_glm_eval
"""
from __future__ import annotations

import logging

from baselines.fnirs.features import FnirsLda
from baselines.fnirs.glm import GlmBeta
from core.data import store
from core.data.fnirs.base import FnirsCfg
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload._eval import CvConfig, CvData, Eval

logger = logging.getLogger(__name__)

_DATASET = "shin2017_nback"
_ARMS = [("collapse (mean+slope+peak)", FnirsLda),
         ("glm-β (HRF, no deriv)", lambda: GlmBeta(derivatives=False)),
         ("glm-β (HRF + derivs)", lambda: GlmBeta(derivatives=True))]


class FnirsGlmEval:
    """GLM-β fNIRS eval helpers — the free helpers folded in as staticmethods."""

    @staticmethod
    def _acc(build, data: CvData, config: CvConfig):
        return Eval.cv_score(build, data, config)[0]


def main():
    Cli.setup_logging()
    meta = store.Store.load(_DATASET, FnirsCfg())                          # clean=None (no filter, as requested)
    X, y = store.Store.gather(meta)
    data = CvData(X, y, meta["subject"].to_numpy())
    logger.info(f"GLM-β vs collapse · Shin n-back · {len(y)} blocks · {meta['subject'].n_unique()} subj · "
          f"3x5-fold · no cleaning\n")
    logger.info(f"  {'arm':<28}{'within':>8}{'cross':>8}   {'0v-load':>8}{'2-v-3':>7}   (3-class chance .333)")
    for name, build in _ARMS:
        wa = FnirsGlmEval._acc(build, data, CvConfig(grouped=False))
        ca = FnirsGlmEval._acc(build, data, CvConfig(grouped=True))
        # 0-back vs 3-back (load on/off proxy)
        ol = FnirsGlmEval._acc(build, data, CvConfig(grouped=True, classes=[0, 2]))
        lv = FnirsGlmEval._acc(build, data, CvConfig(grouped=True, classes=[1, 2]))   # 2-back vs 3-back (level)
        logger.info(f"  {name:<28}{wa:>8.3f}{ca:>8.3f}   {ol:>8.3f}{lv:>7.3f}")
    logger.info("\n  0v-load / 2-v-3 are cross-subject binary (chance 0.5). GLM-β sharpening 0v-load but NOT 2-v-3\n"
          "  = it's physiology, not features (the robust diagnostic).")


if __name__ == "__main__":
    main()
