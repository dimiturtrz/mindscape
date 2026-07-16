"""Braindecode decoders behind one shared trainer — commodity (EEGNet) up to near-SOTA on BCI IV-2a
(ATCNet, EEGConformer). A "method" is a braindecode model name + training hparams; the trainer (AdamW +
cosine LR, per-channel standardization, **sliding-window crop augmentation**, early stopping) is shared.

Crop augmentation is the critical trick behind published 2a results: each ~T/2 trial is cut into
many overlapping sub-windows, turning ~288 trials into thousands of training samples; at test, the crops
of a trial are predicted and their softmax averaged. Without it, deep nets lose to CSP+LDA on 2a's tiny
per-subject sets. The model is still commodity — the contribution is the harness + robust comparison.

Interface = the harness contract: `make(method) -> (fit_fn, score_fn)`. Each trained module is
ONNX-exportable (the Stage-2 edge path rides on it). RTX 5090 + bf16 autocast: minutes from scratch.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from braindecode.models import ATCNet, Deep4Net, EEGConformer, EEGNetv4, ShallowFBCSPNet
from pydantic import BaseModel

# standardizers + crops live in transforms.py (independently testable); re-exported under the legacy
# private names so callers (e.g. tasks/motor_imagery/quantize.py) keep working.
from neuroscan.models.transforms import Transforms

# method -> (braindecode class, training + crop hparams). crop_frac=0.5 + 16 train crops is the standard
# 2a recipe; strong nets get more epochs (cheap on the 5090, capped by early stopping).
# Per-model recipe. `standardize`: "ems" = exponential-moving standardization (braindecode-canonical 2a
# preprocessing); `crop_frac`: None = feed the FULL trial (for nets with their OWN internal window
# augmentation — ATCNet, EEGConformer); 0.5 = external 2s sliding-window crops (for nets without it).
# `cls` is the braindecode class itself (imported above), not a name to look up — a typo is an import error
# at module load, and `make()` validates the method key, so no separate string→class registry is needed.
MODELS: dict[str, dict] = {
    "eegnet":        {"cls": EEGNetv4,        "epochs": 750, "lr": 1e-3,   "batch": 128,
                      "patience": 80, "crop_frac": 0.5,  "standardize": "ems"},
    "shallow_fbcsp": {"cls": ShallowFBCSPNet, "epochs": 750, "lr": 6.5e-4, "batch": 128,
                      "patience": 80, "crop_frac": 0.5,  "standardize": "ems"},
    "deep4":         {"cls": Deep4Net,        "epochs": 750, "lr": 1e-3,   "batch": 128,
                      "patience": 80, "crop_frac": 0.5,  "standardize": "ems"},
    "atcnet":        {"cls": ATCNet,          "epochs": 750, "lr": 1e-3,   "batch": 128,
                      "patience": 80, "crop_frac": None, "standardize": "ems"},
    "eegconformer":  {"cls": EEGConformer,    "epochs": 750, "lr": 1e-3,   "batch": 128,
                      "patience": 80, "crop_frac": None, "standardize": "ems"},
}
DEFAULTS = {"n_train_crops": 16, "n_test_crops": 8, "log_every": 100, "val_frac": 0.2,
            "crop_frac": 0.5, "standardize": "ems"}

_MIN_TRIALS_FOR_VAL = 8   # need more than this many trials before carving a held-out val split for early stopping


logger = logging.getLogger(__name__)

_crops = Transforms.crops
_standardizer = Transforms.standardizer


@dataclass
class Arch:
    """The braindecode model class + the data-derived input/output shape (channels, time samples the net
    consumes, class count)."""
    cls: type[torch.nn.Module]
    n_chans: int
    n_times: int
    n_classes: int


class TrainCfg(BaseModel):
    """Trainer hyperparameters for a BraindecodeClf. `epochs`/`lr`/`batch`/`weight_decay` = the optimizer loop;
    `device` (None -> auto cuda/cpu); `log_every`/`val_frac`/`patience` = logging + early-stop; `crop_len`
    (None = full trial) + `n_train_crops`/`n_test_crops` = the sliding-window crop augmentation; `standardize`
    = the input standardizer; `seed` varies net init for seed-averaging."""
    model_config = {"arbitrary_types_allowed": True}
    epochs: int
    lr: float
    batch: int = 128
    weight_decay: float = 1e-4
    device: str | None = None
    log_every: int = 0
    val_frac: float = 0.2
    patience: int = 0
    crop_len: int | None = None
    n_train_crops: int = 16
    n_test_crops: int = 8
    standardize: str = "ems"
    seed: int = 0


class BraindecodeClf:
    """A braindecode net wrapped as a Decoder (see core.decoder.Decoder): `fit(X, y) -> self`
    and `predict_proba(X) -> probs`, the same contract the classical baselines satisfy — so the harness
    runs nets and baselines through one path."""

    @staticmethod
    def _take(Xs, y, idx, cl, n_crops):
        """Gather (X, y) for a set of trial indices — cropping into `cl`-length windows when cl is set.
        idx=None (no validation split) returns (None, None), so the caller needs no use_val branch."""
        if idx is None:
            return None, None
        if not cl:
            return Xs[idx], y[idx]
        Xc, cmap = _crops(Xs[idx], cl, n_crops)
        return Xc, y[idx][cmap]

    def __init__(self, arch: Arch, config: TrainCfg):
        self.cfg = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs, self.lr, self.batch, self.wd = config.epochs, config.lr, config.batch, config.weight_decay
        self.log_every, self.val_frac, self.patience = config.log_every, config.val_frac, config.patience
        self.crop_len, self.n_train_crops, self.n_test_crops = (config.crop_len, config.n_train_crops,
                                                                config.n_test_crops)
        self.seed = config.seed
        self.std = _standardizer(config.standardize)
        torch.manual_seed(config.seed)                   # vary net init across seeds (seed-averaging)
        # net consumes crop_len samples when cropping, else the full trial
        self.net = arch.cls(n_chans=arch.n_chans, n_outputs=arch.n_classes, n_times=arch.n_times).to(self.device)

    def _make_train_val(self, Xs, y):
        """Trial-level train/val split (val carved from held-out TRIALS, no crop leakage), then crop."""
        cl = self.crop_len
        use_val = self.patience > 0 and len(Xs) > _MIN_TRIALS_FOR_VAL
        if use_val:
            order = np.random.default_rng(self.seed).permutation(len(Xs))
            nv = max(2, int(len(Xs) * self.val_frac))
            vi, ti = order[:nv], order[nv:]
        else:
            ti, vi = np.arange(len(Xs)), None

        Xtr, ytr = BraindecodeClf._take(Xs, y, ti, cl, self.n_train_crops)
        Xva, yva = BraindecodeClf._take(Xs, y, vi, cl, self.n_test_crops)     # vi is None when no val -> (None, None)
        return Xtr, ytr, Xva, yva

    @staticmethod
    def _enable_fast_matmul(device: str) -> None:
        """TF32 for the residual fp32 matmuls (bd 62ak: ~-22% step, the win is in backward) — parity-safe since the
        loop already runs bf16 autocast, so TF32's 10-bit mantissa is more precise than the bf16 already in use."""
        if device != "cuda":
            return
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def fit(self, X, y):
        BraindecodeClf._enable_fast_matmul(self.device)
        Xs = self.std.fit(X)(X)
        Xtr, ytr, Xva, yva = self._make_train_val(Xs, y)
        xt = torch.tensor(Xtr, device=self.device)
        yt = torch.tensor(ytr, dtype=torch.long, device=self.device)
        use_val = Xva is not None
        if use_val:
            xv = torch.tensor(Xva, device=self.device)
            yv = torch.tensor(yva, dtype=torch.long, device=self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = torch.nn.CrossEntropyLoss()
        amp = self.device == "cuda"
        n = len(xt)
        best_loss, best_state, bad = float("inf"), None, 0
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    loss = lossf(self.net(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()

            if use_val:
                self.net.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    vl = float(lossf(self.net(xv), yv))
                if vl < best_loss - 1e-4:
                    best_loss, bad, best_state = vl, 0, copy.deepcopy(self.net.state_dict())
                else:
                    bad += 1
                if self.log_every and (ep + 1) % self.log_every == 0:
                    logger.info(f"    ep {ep + 1}/{self.epochs}  lr {sched.get_last_lr()[0]:.2e}  "
                                f"val {vl:.3f}  (best {best_loss:.3f}, bad {bad})")
                if bad >= self.patience:
                    logger.info(f"    early stop @ ep {ep + 1} (val {best_loss:.3f})")
                    break
            elif self.log_every and (ep + 1) % self.log_every == 0:
                logger.info(f"    ep {ep + 1}/{self.epochs}  lr {sched.get_last_lr()[0]:.2e}")

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _trial_outputs(self, X, *, softmax: bool):
        """Per-trial outputs (probabilities if softmax else logits), crop-averaged when cropping."""
        Xs = self.std(X)
        self.net.eval()
        if self.crop_len:
            Xc, tidx = _crops(Xs, self.crop_len, self.n_test_crops)
            with torch.no_grad():
                o = self.net(torch.tensor(Xc, device=self.device))
                o = (torch.softmax(o, dim=1) if softmax else o).cpu().numpy()
            out = np.zeros((len(Xs), o.shape[1]), dtype=np.float64)
            np.add.at(out, tidx, o)
            return out / self.n_test_crops
        with torch.no_grad():
            o = self.net(torch.tensor(Xs, device=self.device))
            return (torch.softmax(o, dim=1) if softmax else o).cpu().numpy()

    def predict_proba(self, X):
        return self._trial_outputs(X, softmax=True)

    def predict_logits(self, X):
        """Crop-averaged logits per trial — the input temperature scaling calibrates."""
        return self._trial_outputs(X, softmax=False)

    @staticmethod
    def make(method: str):
        """Return (fit_fn, score_fn) for a registered braindecode method (with crop augmentation)."""
        if method not in MODELS:
            raise KeyError(f"unknown decoder {method!r}; have {sorted(MODELS)}")
        cfg = {**DEFAULTS, **MODELS[method]}

        def fit(X, y, **over):
            p = {**cfg, **over}
            T = X.shape[2]
            crop_len = int(p["crop_frac"] * T) if p.get("crop_frac") else None
            n_times = crop_len or T
            arch = Arch(cls=p["cls"], n_chans=X.shape[1], n_times=n_times, n_classes=int(y.max()) + 1)
            config = TrainCfg(
                epochs=p["epochs"], lr=p["lr"], batch=p["batch"],
                log_every=p.get("log_every", 0), val_frac=p.get("val_frac", 0.2),
                patience=p.get("patience", 0), crop_len=crop_len,
                n_train_crops=p["n_train_crops"], n_test_crops=p["n_test_crops"],
                standardize=p.get("standardize", "ems"), seed=p.get("seed", 0))
            return BraindecodeClf(arch, config).fit(X, y)

        def score(clf, X):
            return clf.predict_proba(X)

        return fit, score
