"""Filter-Bank CSP (FBCSP) — the classic strong *Euclidean* motor-imagery baseline (Ang et al. 2012, the
BCI-IV-2a competition winner). The one non-Riemannian classical method here, kept for reference.

Where the Riemannian methods read the whole covariance on its curved manifold, FBCSP is the pre-manifold
state of the art: split the signal into sub-bands (a filter bank), learn CSP spatial filters *per band*
(maximizing the between-class log-variance ratio), keep the strongest CSP features from each, select the
most informative across bands by **mutual information** (MIBIF), and classify with LDA. It has to *learn*
per-band spatial filters to approximate the sensor-mixing invariance the affine-invariant metric gets for
free — which is exactly why it trails tangent-space methods and collapses cross-subject (CSP filters don't
transfer). Included so the table has the cited reference, not because it's expected to win.

Interface = the harness contract: `fit(X, y) -> self`, `predict_proba(X) -> probs[n, C]`. X is [n, ch, t]
at `fs` Hz — the sample rate is needed to *design* the band filters, so it's a constructor arg (as with
the band-power baseline). Give it a broadband recipe (e.g. 4–40 Hz) so the filter bank has range to split.
"""
from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from pydantic import BaseModel
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif

from baselines.base import Baseline


class FbcspConfig(BaseModel):
    """FBCSP hyperparameters. `fs` = epoch sample rate (Hz, needed to design the band filters); the filter
    bank tiles [`fmin`, `fmax`] into `band_width`-wide sub-bands, `order` = Butterworth order, CSP keeps
    `n_components` per band, MI keeps the top `k_features` across all bands."""
    fs: float = 128.0
    fmin: float = 4.0
    fmax: float = 40.0
    band_width: float = 4.0
    order: int = 5
    n_components: int = 4
    k_features: int = 8


class Fbcsp(Baseline):
    """Filter-bank CSP + mutual-info selection + LDA (configured by `FbcspConfig`). Fitted state (per-band
    filters, selected columns, LDA) lives on self."""

    def __init__(self, config: FbcspConfig | None = None):
        self.cfg = config or FbcspConfig()

    def _bands(self) -> list[tuple[float, float]]:
        """Non-overlapping fixed-width sub-bands tiling [fmin, fmax] at band_width (4–8, 8–12, … — the Ang
        2012 bank). Pick fmax **on-grid** (fmin + k·band_width): fixed-width tiling leaves any final remainder
        uncovered — e.g. 4–30 Hz at width 4 tiles up to 24–28 and drops 28–30. That drop is deliberate here:
        it's a 2 Hz high-beta sliver, irrelevant to the theta/alpha workload signal, and covering it would add
        a ragged narrow band + force a re-eval of a baseline that already trails. 2a (4–40) is on-grid."""
        edges = np.arange(self.cfg.fmin, self.cfg.fmax + 1e-6, self.cfg.band_width)
        return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]

    def _sos_bank(self):
        """Butterworth SOS (second-order sections) per band, designed ONCE and cached — the canonical FBCSP
        filter (Ang 2012). scipy's `sosfiltfilt` applies it vectorized over the time axis; going straight to
        scipy (not `mne.filter.filter_data`) drops ~100x of per-call design/padding overhead."""
        if not hasattr(self, "_sos_"):
            self._sos_ = [butter(self.cfg.order, (lo, hi), btype="band", fs=self.cfg.fs, output="sos")
                          for lo, hi in self._bands()]
        return self._sos_

    @staticmethod
    def _filter(X: np.ndarray, sos) -> np.ndarray:
        return sosfiltfilt(sos, X, axis=-1)                       # zero-phase, vectorized over trials×channels

    def _csp(self):
        return CSP(n_components=self.cfg.n_components, reg="ledoit_wolf", log=True)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        sos_bank = self._sos_bank()
        self.csps_, feats = [], []
        for sos in sos_bank:
            csp = self._csp()
            feats.append(csp.fit_transform(self._filter(X, sos), y))      # [n, n_components] log-variance
            self.csps_.append(csp)
        F = np.concatenate(feats, axis=1)                                 # [n, n_bands*n_components]

        # MIBIF: keep the k most class-informative features across the whole bank (skip if we have fewer)
        k = min(self.cfg.k_features, F.shape[1])
        mi = mutual_info_classif(F, y, random_state=0)
        self.sel_ = np.argsort(mi)[::-1][:k]
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[:, self.sel_], y)
        return self

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        feats = [csp.transform(self._filter(X, sos)) for sos, csp in zip(self._sos_bank(), self.csps_, strict=True)]
        return np.concatenate(feats, axis=1)[:, self.sel_]

    def predict_proba(self, X):
        return self.lda_.predict_proba(self._features(X))


def fit(X: np.ndarray, y: np.ndarray, config: FbcspConfig | None = None) -> Baseline:
    """Back-compat shim — prefer `Fbcsp(...).fit(X, y)`."""
    return Fbcsp(config).fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)
