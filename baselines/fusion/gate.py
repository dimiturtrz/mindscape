"""The compact INPUT-level gated fusion model — shallow per-modality encoders + a per-trial gate that mixes
their class-probabilities. Sized for n≈26 / ~700 blocks (d_model 16, dropout ≥ 0.5, weight decay, early
stopping). This is the one fusion path that reads the raw INPUT features rather than the decisions, so it
*could* capture the oracle headroom the output-space combiners can't — measured, it doesn't (see the runner
`tasks/workload/fusion_gate.py`): it ties z-scored-EEG-alone. Kept as a measured null."""
from __future__ import annotations

import torch
from pydantic import BaseModel
from torch import nn

from baselines.fusion.base import FusionData

SEED = 0


class GateConfig(BaseModel):
    """GatedFusion hyperparameters. `eeg_dim`/`fnirs_dim` are the per-modality feature widths (set from the
    data); the rest are the shape + training knobs, kept small/regularized for the tiny fusion set."""
    eeg_dim: int
    fnirs_dim: int
    n_classes: int = 3
    d_model: int = 16
    dropout: float = 0.5
    weight_decay: float = 1e-2
    lr: float = 3e-3
    max_epochs: int = 200
    patience: int = 20


class GatedFusion:
    """Shallow per-modality encoders + a gate that mixes their class-probabilities per trial. Kept tiny and
    heavily regularized; trained with early stopping on an inner validation split."""

    def __init__(self, config: GateConfig):
        self.cfg = config

    def _build(self):
        cfg = self.cfg
        hidden = cfg.d_model

        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                def encoder(in_dim):
                    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(cfg.dropout))

                self.enc_eeg, self.enc_fnirs = encoder(cfg.eeg_dim), encoder(cfg.fnirs_dim)
                self.head_eeg = nn.Linear(hidden, cfg.n_classes)
                self.head_fnirs = nn.Linear(hidden, cfg.n_classes)
                self.gate = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
                                          nn.Linear(hidden, 1))          # per-trial scalar α (pre-sigmoid)

            def forward(self, eeg, fnirs):
                z_eeg, z_fnirs = self.enc_eeg(eeg), self.enc_fnirs(fnirs)
                p_eeg = torch.softmax(self.head_eeg(z_eeg), dim=1)
                p_fnirs = torch.softmax(self.head_fnirs(z_fnirs), dim=1)
                alpha = torch.sigmoid(self.gate(torch.cat([z_eeg, z_fnirs], dim=1)))  # [n,1] in (0,1)
                return alpha * p_eeg + (1 - alpha) * p_fnirs, alpha.squeeze(1)

        return Net()

    def fit(self, train: FusionData, val: FusionData):
        torch.manual_seed(SEED)
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        nll = torch.nn.NLLLoss()
        eeg_tr, fnirs_tr, y_tr = map(torch.as_tensor, (train.eeg, train.fnirs, train.y))
        eeg_va, fnirs_va, y_va = map(torch.as_tensor, (val.eeg, val.fnirs, val.y))
        best, best_state, bad = 1e9, None, 0
        for _ep in range(self.cfg.max_epochs):
            self.net.train()
            opt.zero_grad()
            probs, _ = self.net(eeg_tr, fnirs_tr)
            loss = nll(torch.log(probs + 1e-12), y_tr)
            loss.backward()
            opt.step()
            self.net.eval()
            with torch.no_grad():
                probs_va, _ = self.net(eeg_va, fnirs_va)
                val_loss = nll(torch.log(probs_va + 1e-12), y_va).item()
            if val_loss < best - 1e-4:
                best, best_state, bad = val_loss, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, eeg, fnirs):
        self.net.eval()
        with torch.no_grad():
            probs, alpha = self.net(torch.as_tensor(eeg), torch.as_tensor(fnirs))
        return probs.numpy(), alpha.numpy()
