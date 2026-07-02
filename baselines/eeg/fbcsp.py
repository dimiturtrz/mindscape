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

from baselines.base import Baseline


class Fbcsp(Baseline):
    """Filter-bank CSP + mutual-info selection + LDA. `fs` = epoch sample rate (Hz); the filter bank tiles
    [`fmin`, `fmax`] into `band_width`-wide sub-bands, CSP keeps `n_components` per band, MI keeps the top
    `k_features` across all bands. Fitted state (per-band filters, selected columns, LDA) lives on self."""

    def __init__(self, fs: float = 128.0, fmin: float = 4.0, fmax: float = 40.0, band_width: float = 4.0,
                 order: int = 5, n_components: int = 4, k_features: int = 8):
        self.fs = fs
        self.fmin, self.fmax, self.band_width = fmin, fmax, band_width
        self.order = order
        self.n_components = n_components
        self.k_features = k_features

    def _bands(self) -> list[tuple[float, float]]:
        """Non-overlapping sub-bands tiling [fmin, fmax] at band_width (4–8, 8–12, … — the Ang 2012 bank)."""
        edges = np.arange(self.fmin, self.fmax + 1e-6, self.band_width)
        return [(float(lo), float(hi)) for lo, hi in zip(edges[:-1], edges[1:])]

    def _sos_bank(self):
        """Butterworth SOS (second-order sections) per band, designed ONCE and cached — the canonical FBCSP
        filter (Ang 2012). scipy's `sosfiltfilt` applies it vectorized over the time axis; going straight to
        scipy (not `mne.filter.filter_data`) drops ~100x of per-call design/padding overhead."""
        from scipy.signal import butter
        if not hasattr(self, "_sos_"):
            self._sos_ = [butter(self.order, (lo, hi), btype="band", fs=self.fs, output="sos")
                          for lo, hi in self._bands()]
        return self._sos_

    @staticmethod
    def _filter(X: np.ndarray, sos) -> np.ndarray:
        from scipy.signal import sosfiltfilt
        return sosfiltfilt(sos, X, axis=-1)                       # zero-phase, vectorized over trials×channels

    def _csp(self):
        from mne.decoding import CSP
        return CSP(n_components=self.n_components, reg="ledoit_wolf", log=True)

    def fit(self, X, y):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.feature_selection import mutual_info_classif

        X = np.asarray(X, dtype=np.float64)
        sos_bank = self._sos_bank()
        self.csps_, feats = [], []
        for sos in sos_bank:
            csp = self._csp()
            feats.append(csp.fit_transform(self._filter(X, sos), y))      # [n, n_components] log-variance
            self.csps_.append(csp)
        F = np.concatenate(feats, axis=1)                                 # [n, n_bands*n_components]

        # MIBIF: keep the k most class-informative features across the whole bank (skip if we have fewer)
        k = min(self.k_features, F.shape[1])
        mi = mutual_info_classif(F, y, random_state=0)
        self.sel_ = np.argsort(mi)[::-1][:k]
        self.lda_ = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(F[:, self.sel_], y)
        return self

    def _features(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        feats = [csp.transform(self._filter(X, sos)) for sos, csp in zip(self._sos_bank(), self.csps_)]
        return np.concatenate(feats, axis=1)[:, self.sel_]

    def predict_proba(self, X):
        return self.lda_.predict_proba(self._features(X))


def fit(X: np.ndarray, y: np.ndarray, **kw) -> Baseline:
    """Back-compat shim — prefer `Fbcsp(...).fit(X, y)`."""
    return Fbcsp(**kw).fit(X, y)


def score(clf: Baseline, X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X)
