"""fNIRS feature subset + aggregation study — differentiable, continuous, GPU.

Make the continuous weights actually *select*, by gradient descent instead of black-box search. Per-group
**logits** → **softmax** to a simplex `w` (sum 1, so only the relative split matters — magnitude can't
confound the sparsity term). A linear head classifies the weighted, standardised features; the two train
jointly:

    loss = cross_entropy(head(X_std * w), y)  +  lambda * entropy(w)

`entropy(w)` is the differentiable sparsity penalty — minimising it concentrates weight on the few families
that earn it (good metrics grab weight, junk starves). Sweep `lambda` → the accuracy vs effective-#-features
tradeoff curve (the Pareto analog, no black-box search). The knee subset is validated honestly on a **sealed
subject holdout** the sweep never touches.

`grain='family'` learns one weight per descriptor family (15); `grain='channel'` learns one per column
(72×15=1080) — infeasible for black-box search, trivial here (the reason for torch). Runs on CUDA if present.

    python -m neuroscan.tasks.workload.fnirs_subset_select              # uses fnirs_subset.yaml
    python -m neuroscan.tasks.workload.fnirs_subset_select --grain channel
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.config import REPO
from core.data import store
from core.data.fnirs.base import FnirsCfg
from core.features import extract_bank, family_names

_DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WeightedLinear(torch.nn.Module):
    """Softmax feature weights (per group) × standardised features → linear head. `group_idx[j]` = the
    weight-group of column j (per-family: 0..14; per-channel: j itself), so one learnable logit per group
    broadcasts to its columns."""

    def __init__(self, group_idx: torch.Tensor, n_groups: int, d: int, n_classes: int):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(n_groups))
        self.head = torch.nn.Linear(d, n_classes)
        self.register_buffer("group_idx", group_idx)

    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def entropy(self) -> torch.Tensor:
        w = self.weights()
        return -(w * (w + 1e-12).log()).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x * self.weights()[self.group_idx])


def _fit(Xtr, ytr, group_idx, n_groups, n_classes, lam, hp) -> WeightedLinear:
    model = WeightedLinear(group_idx, n_groups, Xtr.shape[1], n_classes).to(_DEV)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    Xt = torch.as_tensor(Xtr, dtype=torch.float32, device=_DEV)
    yt = torch.as_tensor(ytr, dtype=torch.long, device=_DEV)
    model.train()
    for _ in range(hp["epochs"]):
        opt.zero_grad()
        loss = F.cross_entropy(model(Xt), yt) + lam * model.entropy()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def _predict(model, X) -> np.ndarray:
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32, device=_DEV)).argmax(1).cpu().numpy()


def _standardise(Xtr, Xte):
    """Fit per-feature standardisation on TRAIN, apply to both (no leakage) — the weights act on unit-scale
    features so raw-scale differences between metrics don't bias the weighting."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _cv_acc(F_, group_idx, y, groups, n_groups, n_classes, lam, hp):
    """Mean CV accuracy at this lambda, over repeated seeded StratifiedGroupKFold (subject-grouped)."""
    from sklearn.model_selection import StratifiedGroupKFold
    accs = []
    for seed in hp["fold_seeds"]:
        for tr, te in StratifiedGroupKFold(hp["k"], shuffle=True, random_state=seed).split(F_, y, groups):
            Xtr, Xte = _standardise(F_[tr], F_[te])
            m = _fit(Xtr, y[tr], group_idx, n_groups, n_classes, lam, hp)
            accs.append(float((_predict(m, Xte) == y[te]).mean()))
    return float(np.mean(accs))


def _effective_n(w: np.ndarray) -> float:
    p = w[w > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def _knee(points):
    """Utopia-corner knee of the (acc, eff_n) sweep: closest to high-acc / low-eff_n."""
    acc = np.array([p["acc"] for p in points]); en = np.array([p["eff_n"] for p in points])
    an = (acc - acc.min()) / (np.ptp(acc) + 1e-9)
    en_ = (en - en.min()) / (np.ptp(en) + 1e-9)
    return points[int(np.hypot(1 - an, en_).argmin())]


def _family_weights(w, group_idx_np, families, grain):
    """Report weights per family: identity for grain=family, else sum the column weights within each family."""
    if grain == "family":
        return {f: float(w[i]) for i, f in enumerate(families)}
    ch = len(group_idx_np) // len(families)
    return {f: float(w[i * ch:(i + 1) * ch].sum()) for i, f in enumerate(families)}


def main():
    from omegaconf import OmegaConf
    from sklearn.model_selection import GroupShuffleSplit

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="fnirs_subset.yaml")
    ap.add_argument("--grain", default=None, choices=["family", "channel"])
    args = ap.parse_args()

    cfg = OmegaConf.load(REPO / args.config)
    grain = args.grain or cfg.grain
    hp = {"lr": cfg.lr, "weight_decay": cfg.weight_decay, "epochs": cfg.epochs,
          "k": cfg.k, "fold_seeds": list(cfg.fold_seeds)}

    meta = store.load(cfg.dataset, FnirsCfg())
    X, y = store.gather(meta)
    groups = meta["subject"].to_numpy()
    Fb, fam = extract_bank(X)
    families = family_names()
    n_classes = int(y.max()) + 1

    # weight-group per column: family id (grain=family) or the column itself (grain=channel)
    fam_to_id = {f: i for i, f in enumerate(families)}
    gi_np = np.array([fam_to_id[f] for f in fam]) if grain == "family" else np.arange(Fb.shape[1])
    n_groups = len(families) if grain == "family" else Fb.shape[1]
    group_idx = torch.as_tensor(gi_np, dtype=torch.long, device=_DEV)

    search_idx, seal_idx = next(GroupShuffleSplit(1, test_size=cfg.holdout_frac,
                                                  random_state=cfg.holdout_seed).split(Fb, y, groups))
    Fs, ys, gs = Fb[search_idx], y[search_idx], groups[search_idx]
    print(f"fNIRS subset (torch/{_DEV.type}): {Fb.shape[0]} blocks · grain {grain} ({n_groups} weights) · "
          f"search {len(np.unique(gs))} / sealed {len(np.unique(groups[seal_idx]))} subj · "
          f"lambdas {list(cfg.lambdas)} (chance {1/n_classes:.3f})")

    sweep = []
    for lam in cfg.lambdas:
        acc = _cv_acc(Fs, group_idx, ys, gs, n_groups, n_classes, float(lam), hp)
        Fs_std, _ = _standardise(Fs, Fs)                          # weights read from a full-search fit
        w_full = _fit(Fs_std, ys, group_idx, n_groups, n_classes, float(lam), hp)
        w = w_full.weights().detach().cpu().numpy()
        fw = _family_weights(w, gi_np, families, grain)
        sweep.append({"lam": float(lam), "acc": acc, "eff_n": _effective_n(w), "family_weights": fw})
        top = sorted(fw.items(), key=lambda kv: -kv[1])[:4]
        print(f"  λ={float(lam):<5} acc {acc:.3f} · eff-#feat {sweep[-1]['eff_n']:.2f} · "
              f"top {[(f, round(v, 2)) for f, v in top]}")

    knee = _knee(sweep)
    # honest: refit at the knee lambda on ALL search subjects, score the sealed holdout
    Xtr, Xte = _standardise(Fs, Fb[seal_idx])
    m = _fit(Xtr, ys, group_idx, n_groups, n_classes, knee["lam"], hp)
    sealed = float((_predict(m, Xte) == y[seal_idx]).mean())
    kept = {f: round(v, 3) for f, v in sorted(knee["family_weights"].items(), key=lambda kv: -kv[1]) if v > 0.05}
    print(f"\nknee λ={knee['lam']}: eff-#feat {knee['eff_n']:.2f} · search-acc {knee['acc']:.3f} (optimistic) "
          f"· SEALED-acc {sealed:.3f} (honest)")
    print(f"knee subset (family weight>0.05): {kept}")

    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"subset_{grain}.json").write_text(json.dumps(
        {"dataset": str(cfg.dataset), "grain": grain, "sweep": sweep,
         "knee": knee, "sealed_acc": sealed, "kept": kept}, indent=2))
    print(f"-> {out}/subset_{grain}.json")


if __name__ == "__main__":
    main()
