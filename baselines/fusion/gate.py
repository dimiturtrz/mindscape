"""The compact INPUT-level gated fusion model — shallow per-modality encoders + a per-trial gate that mixes
their class-probabilities. Sized for n≈26 / ~700 blocks (d_model 16, dropout ≥ 0.5, weight decay, early
stopping). This is the one fusion path that reads the raw INPUT features rather than the decisions, so it
*could* capture the oracle headroom the output-space combiners can't — measured, it doesn't (see the runner
`tasks/workload/fusion_gate.py`): it ties z-scored-EEG-alone. Kept as an honest negative."""
from __future__ import annotations

import numpy as np

SEED = 0


class GatedFusion:
    """Shallow per-modality encoders + a gate that mixes their class-probabilities per trial. Kept tiny and
    heavily regularized; trained with early stopping on an inner validation split."""

    def __init__(self, d_e, d_f, n_classes=3, d_model=16, dropout=0.5, wd=1e-2, lr=3e-3, max_epochs=200,
                 patience=20):
        self.cfg = dict(d_e=d_e, d_f=d_f, n_classes=n_classes, d_model=d_model, dropout=dropout,
                        wd=wd, lr=lr, max_epochs=max_epochs, patience=patience)

    def _build(self):
        import torch.nn as nn

        c = self.cfg
        d = c["d_model"]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                enc = lambda din: nn.Sequential(nn.Linear(din, d), nn.ReLU(), nn.Dropout(c["dropout"]))
                s.enc_e, s.enc_f = enc(c["d_e"]), enc(c["d_f"])
                s.head_e = nn.Linear(d, c["n_classes"])
                s.head_f = nn.Linear(d, c["n_classes"])
                s.gate = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(c["dropout"]),
                                       nn.Linear(d, 1))          # per-trial scalar α (pre-sigmoid)

            def forward(s, xe, xf):
                import torch
                ze, zf = s.enc_e(xe), s.enc_f(xf)
                pe = torch.softmax(s.head_e(ze), dim=1)
                pf = torch.softmax(s.head_f(zf), dim=1)
                a = torch.sigmoid(s.gate(torch.cat([ze, zf], dim=1)))  # [n,1] in (0,1)
                p = a * pe + (1 - a) * pf
                return p, a.squeeze(1)

        return Net()

    def fit(self, Xe, Xf, y, Xe_va, Xf_va, y_va):
        import torch

        torch.manual_seed(SEED)
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        nll = torch.nn.NLLLoss()
        te, tf, ty = map(torch.as_tensor, (Xe, Xf, y))
        ve, vf, vy = map(torch.as_tensor, (Xe_va, Xf_va, y_va))
        best, best_state, bad = 1e9, None, 0
        for _ep in range(self.cfg["max_epochs"]):
            self.net.train(); opt.zero_grad()
            p, _ = self.net(te, tf)
            loss = nll(torch.log(p + 1e-12), ty)
            loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                pv, _ = self.net(ve, vf)
                vloss = nll(torch.log(pv + 1e-12), vy).item()
            if vloss < best - 1e-4:
                best, best_state, bad = vloss, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, Xe, Xf):
        import torch

        self.net.eval()
        with torch.no_grad():
            p, a = self.net(torch.as_tensor(Xe), torch.as_tensor(Xf))
        return p.numpy(), a.numpy()
