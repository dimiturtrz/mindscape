"""GLM-β fNIRS decoder — model-based amplitude instead of a hand-collapsed mean/slope/peak.

The field-standard `FnirsLda` crushes each channel's 22 s trajectory to 3 hand-picked scalars. This decoder
keeps the *shape* by fitting the neuroimaging-standard **general linear model** per channel: the block's
hemodynamic response is a canonical HRF (double-gamma) convolved with the task boxcar, and the fitted weight
**β** is *how strongly that trial matches the HRF template* — a temporal model, not a naive average. Two extra
regressors (the HRF's temporal and dispersion derivatives) absorb per-subject variation in response delay and
width, so their βs carry cross-subject-robust shape information a fixed mean can't. Polynomial drift regressors
are nuisances (soak up residual trend), not features.

Features per channel = [β_HRF, β_tderiv, β_disp] (HbO then HbR) → StandardScaler → shrinkage-LDA — same
classifier as the baseline, so a delta isolates the GLM feature. GLM-β beats mean/slope by +7–18 % on
motor/rotation tasks (Uga 2014 / PMC7040364); this is the first cross-subject n-back test of it (open question
in the 2026-07-05 deep-dive). `fs`, `tmin`, `task_dur` set the epoch timing (Shin: 10 Hz, −2 s baseline,
~20 s task within the window).
"""
from __future__ import annotations

import numpy as np

from baselines.base import Baseline


def _canonical_hrf(fs: float, length_s: float = 32.0, peak: float = 6.0, under: float = 16.0,
                   p_disp: float = 1.0, u_disp: float = 1.0, ratio: float = 1 / 6) -> np.ndarray:
    """SPM-style double-gamma HRF sampled at `fs`, peak-normalized. `p_disp`/`u_disp` widen the gammas (used
    to form the dispersion-derivative basis)."""
    from scipy.stats import gamma
    t = np.arange(0, length_s, 1.0 / fs)
    h = gamma.pdf(t, peak / p_disp, scale=p_disp) - ratio * gamma.pdf(t, under / u_disp, scale=u_disp)
    return h / np.abs(h).max()


class GlmBeta(Baseline):
    """Per-channel GLM with a canonical-HRF (+temporal/dispersion-derivative) task regressor; the fitted βs
    are the features. `derivatives=False` keeps only β_HRF (1 feature/channel); `drift_order` sets the number
    of polynomial nuisance drift regressors."""

    def __init__(self, fs: float = 10.0, tmin: float = -2.0, task_dur: float = 20.0,
                 derivatives: bool = True, drift_order: int = 1):
        self.fs = fs
        self.tmin = tmin
        self.task_dur = task_dur
        self.derivatives = derivatives
        self.drift_order = drift_order

    def _design(self, T: int) -> tuple[np.ndarray, int]:
        """Design matrix `[T, K]` and the number of leading FEATURE columns (βs we keep). Columns:
        [HRF, (tderiv, disp)?, drift_0..drift_p]. Drift columns are nuisance — dropped from features."""
        t = self.tmin + np.arange(T) / self.fs                       # epoch time axis (s)
        box = ((t >= 0.0) & (t < self.task_dur)).astype(float)       # task ON over the window

        hrf = _canonical_hrf(self.fs)
        bases = [hrf]
        if self.derivatives:
            tderiv = np.gradient(hrf) * self.fs                      # temporal derivative (delay variation)
            disp = (hrf - _canonical_hrf(self.fs, p_disp=1.01)) / 0.01   # dispersion derivative (width variation)
            bases += [tderiv, disp]
        cols = [np.convolve(box, b)[:T] for b in bases]              # regressor = task ⊛ basis
        n_feat = len(cols)

        tn = np.linspace(-1, 1, T)                                   # polynomial drift nuisances (incl. constant)
        cols += [tn ** p for p in range(self.drift_order + 1)]
        D = np.stack(cols, axis=1)                                   # [T, K]
        return D, n_feat

    def _features(self, X: np.ndarray) -> np.ndarray:
        """`X[n,ch,t]` -> `[n, ch*n_feat]` GLM βs (feature columns only), channel-major to match the baseline."""
        X = np.asarray(X, dtype=np.float64)
        n, ch, T = X.shape
        D, n_feat = self._design(T)
        P = np.linalg.pinv(D)                                        # [K, T]
        B = (X.reshape(n * ch, T) @ P.T)[:, :n_feat]                 # [n*ch, K] -> keep feature βs
        return B.reshape(n, ch * n_feat)

    def _build(self):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

    def fit(self, X, y):
        self.pipe_ = self._build().fit(self._features(X), np.asarray(y))
        return self

    def predict_proba(self, X):
        return self.pipe_.predict_proba(self._features(X))
