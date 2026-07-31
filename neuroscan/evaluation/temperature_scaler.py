"""Post-hoc temperature scaling (Guo 2017) — the calibration operator the `calibrate` runner fits.

One scalar T (logits -> logits/T), fit on a held-out val set by minimizing NLL with the model frozen; T>1
softens overconfidence. It does NOT change argmax, so accuracy is untouched — only confidence (ECE) moves.
"""
from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float, Int

from neuroscan.evaluation import metrics


class TemperatureScaler:
    """Post-hoc temperature scaling (Guo 2017): one scalar T (logits -> logits/T), fit on a held-out val
    set by minimizing NLL with the model frozen. The object OWNS T and the two operations that use it —
    `.fit` sets T, `.ece` reports ECE at the fitted T (or an override). Softmax argmax is unchanged, so
    accuracy is untouched; only confidence (ECE) moves."""

    def __init__(self, T: float = 1.0):
        self.T = T

    def fit(self, logits: Float[np.ndarray, "n c"], labels: Int[np.ndarray, "n"]) -> "TemperatureScaler":
        z = torch.tensor(logits, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.long)
        log_t = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([log_t], lr=0.05, max_iter=80)
        nll = torch.nn.CrossEntropyLoss()

        def closure():                                       # LBFGS requires a closure
            opt.zero_grad()
            loss = nll(z / log_t.exp(), y)
            loss.backward()
            return loss

        opt.step(closure)
        self.T = float(log_t.exp().detach())
        return self

    def probs(self, logits: Float[np.ndarray, "n c"], T: float | None = None) -> Float[np.ndarray, "n c"]:
        """Numerically-stable softmax(logits / T); T defaults to the fitted self.T."""
        z = logits / (self.T if T is None else T)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def ece(self, logits: Float[np.ndarray, "n c"], labels: Int[np.ndarray, "n"], T: float | None = None) -> float:
        p = self.probs(logits, T)
        conf, pred = p.max(1), p.argmax(1)
        return metrics.Metrics.ece(conf, (pred == labels).astype(float))[0]
