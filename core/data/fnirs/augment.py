"""Domain-randomization augmentation for fNIRS cross-subject robustness (bd jdh).

Between-subject fNIRS variation is mostly *nuisance*: optode-coupling gain differences, subject-specific
systemic physiology (Mayer/respiration/cardiac), and small hemodynamic-timing shifts. Domain randomization
injects that nuisance at train time so the decoder learns to ignore it — the augmentation analog of the
per-subject z-scoring / re-centering transfer fixes.

`domain_randomize` applies, per epoch: a log-normal per-channel gain (coupling variability), an added
common-mode systemic burst (reusing the synthetic forward's oscillator, random amplitude/phase), and a small
random temporal shift (hemodynamic-timing jitter). It preserves the neural anti-correlation CBSI reads — it
perturbs the nuisance axes, not the HbO/HbR contrast.

APPLICABILITY (honest, per jdh's "where signal exists"): this helps only a task with cross-subject transfer
HEADROOM. Our fNIRS n-back workload is physiologically ceiling-bound (bd memory) and MI cross-subject is
already closed by Riemannian re-centering — so the mechanism is provided opt-in + documented; a decode gain
should be claimed only on a task shown to have residual nuisance-driven cross-subject loss.
"""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from core.data.fnirs.synthetic import SynthConfig, _systemic


class AugConfig(BaseModel):
    """Domain-randomization strengths — the between-subject nuisance axes to diversify."""
    gain_sigma: float = 0.2        # log-normal per-channel coupling-gain variability
    systemic_amp: float = 0.3      # added common-mode systemic burst amplitude
    max_shift_s: float = 1.0       # circular hemodynamic-timing jitter (s)


def domain_randomize(hbo: np.ndarray, hbr: np.ndarray, fs: float, cfg: AugConfig | None = None,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Randomize the nuisance axes of paired (HbO, HbR) `[n, ch, t]` for cross-subject robustness (bd jdh).

    Per epoch: log-normal per-channel gain (coupling variability), an added common-mode systemic burst, and a
    circular temporal shift (hemodynamic-timing jitter) — strengths from `cfg` (AugConfig). Common-mode terms
    hit HbO and HbR alike, so CBSI still cancels them: the neural contrast is preserved, only the nuisance is
    diversified."""
    cfg = cfg or AugConfig()
    gain_sigma, systemic_amp, max_shift_s = cfg.gain_sigma, cfg.systemic_amp, cfg.max_shift_s
    rng = np.random.default_rng(seed)
    o = np.asarray(hbo, dtype=np.float64)
    r = np.asarray(hbr, dtype=np.float64)
    n, ch, t = o.shape

    gain = rng.lognormal(0.0, gain_sigma, (n, ch, 1))
    sys_cfg = SynthConfig(systemic_amp=systemic_amp)
    systemic = _systemic(n * ch, t, fs, sys_cfg, rng).reshape(n, ch, t)      # common-mode, per channel
    shifts = rng.integers(-int(max_shift_s * fs), int(max_shift_s * fs) + 1, n)

    out_o = np.empty_like(o)
    out_r = np.empty_like(r)
    for i in range(n):
        out_o[i] = np.roll(o[i] * gain[i] + systemic[i], shifts[i], axis=-1)
        out_r[i] = np.roll(r[i] * gain[i] + systemic[i], shifts[i], axis=-1)
    return out_o.astype(np.float32), out_r.astype(np.float32)
