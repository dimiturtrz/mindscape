"""Multi-seed parity test for the NICE perception recipe (bd mindscape-9s5).

The `xam` run showed the optimized batching recipe (balanced / CLIP-hard negatives, hard-neg loss
weighting, bf16, per-epoch train_frac subsampling) scores at near-parity with the naive uniform-fp32
recipe on cross-subject retrieval — at ~5x less compute. Single-seed, the tiny top-1 gap (1.57% vs 1.85%)
is inside plausible seed noise. This runs BOTH arms across several seeds on the SAME split and reports
mean +/- std, so the claim ("the win is efficiency, not accuracy — and it costs no accuracy") rests on a
noise band, not one draw.

    uv run python -m neuroscan.tasks.visual.seed_parity --train 1 2 3 4 --test 5 --seeds 0 1 2

Each arm is a committed config (configs/perception_{naive,optimized}.json); the seed is overridden per run.
Reports single-trial + concept-averaged top-1/5 per arm as mean +/- std and the naive-minus-optimized gap.
"""
import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any, cast

from neuroscan.tasks.cli import Cli
from neuroscan.tasks.visual.train_nice import TrainConfig, TrainNice

logger = logging.getLogger(__name__)

_CFG_DIR = Path(__file__).parent / "configs"
_ARMS = {"naive": "perception_naive.json", "optimized": "perception_optimized.json"}


class SeedParity:
    """Multi-seed parity test for the NICE perception recipe — the free helpers folded in as staticmethods
    (public names kept). `run` trains both arms across seeds; `_agg` reduces the per-seed runs to mean/std."""

    @staticmethod
    def _agg(runs: list[dict[str, Any]], metric: str, k: str) -> dict[str, Any]:
        """mean/std of `runs[i][metric][k]` across seeds (metric = single_trial | concept_avg)."""
        vals = [r[metric][k] for r in runs]
        return {"mean": statistics.fmean(vals), "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "vals": vals}

    @staticmethod
    def run(train_subjects: list[int], test_subject: int, seeds: list[int]) -> dict[str, Any]:
        out: dict[str, Any] = {"train": train_subjects, "test": test_subject, "seeds": seeds,
                               "arms": {}}
        for arm, fname in _ARMS.items():
            base = json.loads((_CFG_DIR / fname).read_text())
            runs = []
            for seed in seeds:
                cfg = TrainConfig(**cast(dict[str, Any], {**base, "seed": seed}))
                logger.info(f"[{arm}] seed {seed} — {fname}")
                runs.append(TrainNice.train(train_subjects, test_subject, cfg))
            out["arms"][arm] = {
                "single_trial": {k: SeedParity._agg(runs, "single_trial", k) for k in ("1", "5")},
                "concept_avg": {k: SeedParity._agg(runs, "concept_avg", k) for k in ("1", "5")},
            }
        n, o = out["arms"]["naive"], out["arms"]["optimized"]
        out["gap_naive_minus_optimized"] = {
            "single_trial_top1": n["single_trial"]["1"]["mean"] - o["single_trial"]["1"]["mean"],
            "concept_avg_top1": n["concept_avg"]["1"]["mean"] - o["concept_avg"]["1"]["mean"],
        }
        return out


def main():
    Cli.setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    result = SeedParity.run(args.train, args.test, args.seeds)
    logger.info(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
