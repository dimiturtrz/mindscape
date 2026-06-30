"""Braindecode decoders behind one shared trainer — commodity (EEGNet) up to near-SOTA on BCI IV-2a
(ATCNet, EEGConformer). A "method" is just a braindecode model name + training hparams; the trainer
(AdamW + cosine LR, per-channel standardization fit on train) is shared. RTX 5090 trains these from
scratch in minutes.

The model is commodity — the contribution is the harness + the honest within→cross-subject comparison,
not any one net. Each trained module is ONNX-exportable as-is (the Stage-2 edge path rides on it).

Interface = the harness contract: `make(method) -> (fit_fn, score_fn)`.
"""
from __future__ import annotations

import numpy as np

# method name -> (braindecode class, training hparams). Hparams are reasonable 2a-4class defaults;
# the strong nets (atcnet/eegconformer) want more epochs — cheap on the 5090.
MODELS: dict[str, dict] = {
    "eegnet":        {"cls": "EEGNetv4",        "epochs": 500,  "lr": 1e-3, "batch": 64},
    "shallow_fbcsp": {"cls": "ShallowFBCSPNet", "epochs": 500,  "lr": 6.5e-4, "batch": 64},
    "deep4":         {"cls": "Deep4Net",        "epochs": 500,  "lr": 1e-3, "batch": 64},
    "atcnet":        {"cls": "ATCNet",          "epochs": 1000, "lr": 1e-3, "batch": 64},
    "eegconformer":  {"cls": "EEGConformer",    "epochs": 1000, "lr": 1e-3, "batch": 72},
}


class _Standardizer:
    """Per-channel z-score, fit on train (mean/std over epochs+time per channel)."""
    def fit(self, X):
        self.mu = X.mean(axis=(0, 2), keepdims=True)
        self.sd = X.std(axis=(0, 2), keepdims=True) + 1e-6
        return self

    def __call__(self, X):
        return ((X - self.mu) / self.sd).astype(np.float32)


class BraindecodeClf:
    def __init__(self, cls: str, n_chans: int, n_times: int, n_classes: int,
                 epochs: int, lr: float, batch: int = 64, weight_decay: float = 1e-4,
                 device: str | None = None, log_every: int = 0):
        import torch
        import braindecode.models as M

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs, self.lr, self.batch, self.wd = epochs, lr, batch, weight_decay
        self.log_every = log_every
        self.std = _Standardizer()
        Net = getattr(M, cls)
        self.net = Net(n_chans=n_chans, n_outputs=n_classes, n_times=n_times).to(self.device)

    def fit(self, X, y):
        import torch

        xt = torch.tensor(self.std.fit(X)(X), device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = torch.nn.CrossEntropyLoss()
        n = len(xt)
        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.net(xt[idx]), yt[idx]).backward()
                opt.step()
            sched.step()
            if self.log_every and (ep + 1) % self.log_every == 0:
                print(f"    ep {ep + 1}/{self.epochs}  lr {sched.get_last_lr()[0]:.2e}")
        return self

    def predict_proba(self, X):
        import torch

        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(self.std(X), device=self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()


def make(method: str):
    """Return (fit_fn, score_fn) for a registered braindecode method."""
    if method not in MODELS:
        raise KeyError(f"unknown decoder {method!r}; have {sorted(MODELS)}")
    cfg = MODELS[method]

    def fit(X, y, **over):
        p = {**cfg, **over}
        return BraindecodeClf(p["cls"], X.shape[1], X.shape[2], int(y.max()) + 1,
                              epochs=p["epochs"], lr=p["lr"], batch=p["batch"]).fit(X, y)

    def score(clf, X):
        return clf.predict_proba(X)

    return fit, score
