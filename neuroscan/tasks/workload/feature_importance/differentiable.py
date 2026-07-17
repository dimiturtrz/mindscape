"""fNIRS feature subset + aggregation study — differentiable, continuous, GPU.

Make the continuous weights actually *select*, by gradient descent instead of black-box search. Per-group
**logits** → **softmax** to a simplex `w` (sum 1, so only the relative split matters — magnitude can't
confound the sparsity term). A linear head classifies the weighted, standardised features; the two train
jointly:

    loss = cross_entropy(head(X_std * w), y)  +  lambda * entropy(w)

`entropy(w)` is the differentiable sparsity penalty — minimising it concentrates weight on the few families
that earn it (good metrics grab weight, junk starves). Sweep `lambda` → the accuracy vs effective-#-features
tradeoff curve (the Pareto analog, no black-box search). The knee subset is validated on a **sealed
subject holdout** the sweep never touches.

`grain='family'` learns one weight per descriptor family (15); `grain='channel'` learns one per column
(72×15=1080) — infeasible for black-box search, trivial here (the reason for torch). Runs on CUDA if present.

    python -m neuroscan.tasks.workload.feature_importance.differentiable              # uses subset.yaml (local)
    python -m neuroscan.tasks.workload.feature_importance.differentiable --grain channel
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from omegaconf import OmegaConf
from sklearn.model_selection import GroupShuffleSplit
from torch import Tensor

from core.config import REPO
from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.features import DescriptorBank
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload.feature_importance._cv import Cv

logger = logging.getLogger(__name__)

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_EPS = 1e-12            # drop numerically-zero simplex weights before the entropy / effective-#-features sum
_KEEP_WEIGHT_MIN = 0.05  # family weight above this is reported in the knee subset (the kept features)
_KEEP_WEIGHT_DP = 3      # decimal places the kept-subset weights are rounded to for the report
_CFG = Path(__file__).with_name("subset.yaml")            # study config lives beside the code (config-as-data)


class WeightedLinear(torch.nn.Module):
    """Softmax feature weights (per group) × standardised features → linear head. `group_idx[j]` = the
    weight-group of column j (per-family: 0..14; per-channel: j itself), so one learnable logit per group
    broadcasts to its columns."""

    def __init__(self, group_idx: Int[Tensor, "f"], n_groups: int, d: int, n_classes: int):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n_groups))
        self.head = torch.nn.Linear(d, n_classes)
        self.register_buffer("group_idx", group_idx)

    def weights(self) -> Float[Tensor, "f"]:
        return torch.softmax(self.logits, dim=0)

    def entropy(self) -> Float[Tensor, ""]:
        w = self.weights()
        return -(w * (w + 1e-12).log()).sum()

    def forward(self, x: Float[Tensor, "n f"]) -> Float[Tensor, "n c"]:
        return self.head(x * self.weights()[self.group_idx])


@dataclass
class GroupSpec:
    """The feature-weighting structure the WeightedLinear model is built from: which weight-group each column
    belongs to (`group_idx`), the number of groups, and the class count."""
    group_idx: torch.Tensor
    n_groups: int
    n_classes: int


@dataclass
class _SearchData:
    """The search-fold data: the feature bank `F`, class labels `y`, and per-block subject `groups`."""
    F: np.ndarray
    y: np.ndarray
    groups: np.ndarray


class Differentiable:
    """The differentiable fNIRS feature-subset study helpers (free functions folded in as staticmethods,
    public names kept)."""

    @classmethod
    def _fit(cls, Xtr, ytr, spec: GroupSpec, lam, hp) -> WeightedLinear:
        model = WeightedLinear(spec.group_idx, spec.n_groups, Xtr.shape[1], spec.n_classes).to(_DEV)
        opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=_DEV)
        yt = torch.as_tensor(ytr, dtype=torch.long, device=_DEV)
        norm = math.log(max(spec.n_groups, 2))                       # normalise entropy by log K so lambda is
        # grain-invariant (same meaning at 15 or 1080 weights)
        model.train()
        for _ in range(hp["epochs"]):
            opt.zero_grad()
            loss = F.cross_entropy(model(Xt), yt) + lam * model.entropy() / norm
            loss.backward()
            opt.step()
        return model

    @classmethod
    @torch.no_grad()
    def _predict(cls, model, X) -> Int[np.ndarray, "n"]:
        model.eval()
        return model(torch.as_tensor(X, dtype=torch.float32, device=_DEV)).argmax(1).cpu().numpy()

    @classmethod
    def _standardise(cls, Xtr, Xte):
        """Fit per-feature standardisation on TRAIN, apply to both (no leakage) — the weights act on unit-scale
        features so raw-scale differences between metrics don't bias the weighting."""
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        return (Xtr - mu) / sd, (Xte - mu) / sd

    @classmethod
    def _cv_acc(cls, data: _SearchData, spec: GroupSpec, lam, hp):
        """Mean CV accuracy at this lambda, over repeated seeded StratifiedGroupKFold (subject-grouped)."""
        accs = []
        for tr, te in Cv.grouped_folds(data.F, data.y, data.groups, hp["fold_seeds"], hp["k"]):
            Xtr, Xte = cls._standardise(data.F[tr], data.F[te])
            m = cls._fit(Xtr, data.y[tr], spec, lam, hp)
            accs.append(float((cls._predict(m, Xte) == data.y[te]).mean()))
        return float(np.mean(accs))

    @classmethod
    def _effective_n(cls, w: Float[np.ndarray, "f"]) -> float:
        p = w[w > _EPS]
        return float(np.exp(-(p * np.log(p)).sum()))

    @classmethod
    def _knee(cls, points):
        """Utopia-corner knee of the (acc, eff_n) sweep: closest to high-acc / low-eff_n."""
        acc = np.array([p["acc"] for p in points])
        en = np.array([p["eff_n"] for p in points])
        an = (acc - acc.min()) / (np.ptp(acc) + 1e-9)
        en_ = (en - en.min()) / (np.ptp(en) + 1e-9)
        return points[int(np.hypot(1 - an, en_).argmin())]

    @classmethod
    def _family_weights(cls, w, group_idx_np, families, grain):
        """Report weights per family: identity for grain=family, else sum the column weights within each family."""
        if grain == "family":
            return {f: float(w[i]) for i, f in enumerate(families)}
        ch = len(group_idx_np) // len(families)
        return {f: float(w[i * ch:(i + 1) * ch].sum()) for i, f in enumerate(families)}

    @classmethod
    def main(cls):
        Cli.setup_logging()
        ap = argparse.ArgumentParser(description=__doc__)
        ap.add_argument("--config", default=None, help="study config (default: subset.yaml beside this module)")
        ap.add_argument("--grain", default=None, choices=["family", "channel"])
        args = ap.parse_args()

        cfg = OmegaConf.load(args.config or _CFG)
        grain = args.grain or cfg.grain
        hp = {"lr": cfg.lr, "weight_decay": cfg.weight_decay, "epochs": cfg.epochs,
              "k": cfg.k, "fold_seeds": list(cfg.fold_seeds)}

        meta = store.Store.load(cfg.dataset, FnirsCfg())
        X, y = store.Store.gather(meta)
        groups = meta["subject"].to_numpy()
        Fb, fam = DescriptorBank.extract_bank(X)
        families = DescriptorBank.family_names()
        n_classes = int(y.max()) + 1

        # weight-group per column: family id (grain=family) or the column itself (grain=channel)
        fam_to_id = {f: i for i, f in enumerate(families)}
        gi_np = np.array([fam_to_id[f] for f in fam]) if grain == "family" else np.arange(Fb.shape[1])
        n_groups = len(families) if grain == "family" else Fb.shape[1]
        group_idx = torch.as_tensor(gi_np, dtype=torch.long, device=_DEV)
        spec = GroupSpec(group_idx, n_groups, n_classes)

        search_idx, seal_idx = next(GroupShuffleSplit(1, test_size=cfg.holdout_frac,
                                                      random_state=cfg.holdout_seed).split(Fb, y, groups))
        Fs, ys, gs = Fb[search_idx], y[search_idx], groups[search_idx]
        search = _SearchData(Fs, ys, gs)
        logger.info(f"fNIRS subset (torch/{_DEV.type}): {Fb.shape[0]} blocks · grain {grain} ({n_groups} weights) · "
              f"search {len(np.unique(gs))} / sealed {len(np.unique(groups[seal_idx]))} subj · "
              f"lambdas {list(cfg.lambdas)} (chance {1/n_classes:.3f})")

        sweep = []
        for lam in cfg.lambdas:
            acc = cls._cv_acc(search, spec, float(lam), hp)
            Fs_std, _ = cls._standardise(Fs, Fs)           # weights read from a full-search fit
            w_full = cls._fit(Fs_std, ys, spec, float(lam), hp)
            w = w_full.weights().detach().cpu().numpy()
            fw = cls._family_weights(w, gi_np, families, grain)
            sweep.append({"lam": float(lam), "acc": acc, "eff_n": cls._effective_n(w), "family_weights": fw})
            top = sorted(fw.items(), key=lambda kv: -kv[1])[:4]
            logger.info(f"  λ={float(lam):<5} acc {acc:.3f} · eff-#feat {sweep[-1]['eff_n']:.2f} · "
                  f"top {[(feature_family, round(weight, 2)) for feature_family, weight in top]}")

        knee = cls._knee(sweep)
        # leakage-free: refit at the knee lambda on ALL search subjects, score the sealed holdout
        Xtr, Xte = cls._standardise(Fs, Fb[seal_idx])
        m = cls._fit(Xtr, ys, spec, knee["lam"], hp)
        sealed = float((cls._predict(m, Xte) == y[seal_idx]).mean())
        kept = {feature_family: round(weight, _KEEP_WEIGHT_DP)
                for feature_family, weight in sorted(knee["family_weights"].items(), key=lambda kv: -kv[1])
                if weight > _KEEP_WEIGHT_MIN}
        logger.info(f"\nknee λ={knee['lam']}: eff-#feat {knee['eff_n']:.2f} · search-acc {knee['acc']:.3f} "
              f"· SEALED-acc {sealed:.3f} (unbiased)")
        logger.info(f"knee subset (family weight>0.05): {kept}")

        out = REPO / cfg.out
        out.mkdir(parents=True, exist_ok=True)
        (out / f"subset_{grain}.json").write_text(json.dumps(
            {"dataset": str(cfg.dataset), "grain": grain, "sweep": sweep,
             "knee": knee, "sealed_acc": sealed, "kept": kept}, indent=2))
        logger.info(f"-> {out}/subset_{grain}.json")


if __name__ == "__main__":
    Differentiable.main()
