"""Windowed stage-1 fNIRS decoder — stop collapsing the hemodynamic trajectory to a single scalar/channel.

The field-standard `FnirsLda` (features.py) takes ONE mean/slope/peak per channel over the whole ~22 s
window. This decoder keeps the time axis: it slices the window into overlapping sub-windows, extracts
per-channel local descriptors in each (a short slope reads the *local* rate of change), then combines the
sub-windows into a block decision one of four ways (`aggregate`). The point is that "keep time" is not one
method — the aggregation is the design axis, and the two obvious endpoints are both pathological, so both
must be tried, not just one:

  - `concat` — stack the sub-windows IN ORDER into one block vector -> shrinkage-LDA. Position-aware (a weight
    per (window, family, channel)), but `W·K·ch`-dimensional: on 702 blocks it overfits, worst cross-subject.
  - `mean`   — a shared low-dim stage-1 LDA scores each sub-window; average the per-window probabilities.
    Few params (transfers well) but order-blind AND it dilutes: a couple of informative windows drown under
    the flat ones.
  - `max`    — same shared stage-1, but pool the per-window class scores by MAX (Multiple-Instance Learning:
    "the block is high-load if ANY sub-window looks high-load"). Keeps a localized cue instead of diluting it.
  - `lse`    — log-sum-exp pool (a soft, differentiable max) — max's robustness without betting on one window.

`mean`/`max`/`lse` share ONE stage-1 across all windows (K·ch params, +1 if `add_position`), so they stay in
the transfer-friendly low-dim regime that `concat` blows past; `add_position` appends the normalized
sub-window centre so the shared stage-1 isn't fully position-blind. The windowing is fully internal:
`fit(X[n,ch,t], y[n])` / `predict_proba(X[n,ch,t]) -> [n, C]` keep the exact harness contract, so subject-
grouped CV never sees the per-window expansion (a block's sub-windows share its label + subject). Stage-1 is
the same shrinkage-LDA as the collapse baseline, so a within/cross delta isolates the representation +
aggregation, nothing else.
"""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines.base import Baseline
from core.features import FNIRS_FEATURE_FNS

# per-sub-window descriptors: local level + local trend. The temporal resolution comes from the number of
# sub-windows, not from a wide per-window bank, so this stays compact (small K).
_DEFAULT_FAMILIES = ("mean", "slope")
_AGGREGATES = ("concat", "mean", "max", "lse")
_MIL_MIN_CLASSES = 3          # MIL score-pooling needs a per-class score axis; binary LDA has only 1 margin
_MULTICLASS_SCORE_NDIM = 2    # multiclass LDA.decision_function is 2-D [n, C]; binary is 1-D


class WindowedFnirs(Baseline):
    """Sub-window the hemodynamic response and combine the windows into a block decision via `aggregate`
    (`concat`|`mean`|`max`|`lse`). `win_s`/`hop_s` = sub-window length and stride in seconds; `fs` converts to
    samples (Shin fNIRS = 10 Hz); `families` = per-window descriptors; `add_position` appends the normalized
    sub-window centre to the pooled stage-1's features (ignored by `concat`, which is position-aware already)."""

    # defaults = the Pareto point from the granularity sweep (neuroscan/tasks/workload/fnirs_windowed_eval.py):
    # 3 coarse ordered windows via `concat` gave the within-subject gain (+~2.5 pp vs collapse) at ~zero
    # cross-subject cost, where finer windows overfit transfer and the pooled modes underperformed collapse.
    def __init__(self, win_s: float = 7.0, hop_s: float = 7.0, fs: float = 10.0,
                 families: tuple[str, ...] = _DEFAULT_FAMILIES, aggregate: str = "concat",
                 *, add_position: bool = True):
        if aggregate not in _AGGREGATES:
            raise ValueError(f"aggregate must be one of {_AGGREGATES}, got {aggregate!r}")
        self.win_s = win_s
        self.hop_s = hop_s
        self.fs = fs
        self.families = tuple(families)
        self.aggregate = aggregate
        self.add_position = add_position

    # --- windowing -------------------------------------------------------------------------------------
    def _starts(self, t: int) -> list[int]:
        """Sub-window start indices for a length-`t` block (clamped so a short block still yields ≥1)."""
        L = max(1, int(round(self.win_s * self.fs)))
        hop = max(1, int(round(self.hop_s * self.fs)))
        if t <= L:
            return [0]
        return list(range(0, t - L + 1, hop))

    def _window_feats(self, X: np.ndarray) -> np.ndarray:
        """`X[n,ch,t]` -> `Fw[n, W, D]`: per-sub-window descriptor vector (D = K·ch, +1 if add_position). D is
        the same across windows, so the pooled modes share one stage-1 over all of them; `concat` flattens."""
        L = max(1, int(round(self.win_s * self.fs)))
        starts = self._starts(X.shape[2])
        T = X.shape[2]
        mats = []
        for s in starts:
            win = X[:, :, s:s + L]                                        # [n, ch, L]
            feats = np.concatenate([np.asarray(FNIRS_FEATURE_FNS[f](win), dtype=np.float64)
                                    for f in self.families], axis=1)      # [n, K*ch]
            if self.add_position and self.aggregate != "concat":
                pos = np.full((feats.shape[0], 1), (s + L / 2) / T)       # normalized sub-window centre
                feats = np.concatenate([feats, pos], axis=1)
            mats.append(feats)
        return np.stack(mats, axis=1)                                     # [n, W, D]

    # --- classifier ------------------------------------------------------------------------------------
    @staticmethod
    def _lda():
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

    def fit(self, X, y):
        Fw = self._window_feats(np.asarray(X))                           # [n, W, D]
        y = np.asarray(y)
        if self.aggregate == "concat":
            self.pipe_ = self._lda().fit(Fw.reshape(len(y), -1), y)      # one high-dim vector per block
        else:
            n, W, D = Fw.shape
            yw = np.repeat(y, W)                                          # each sub-window inherits its block label
            self.pipe_ = self._lda().fit(Fw.reshape(n * W, D), yw)       # ONE shared stage-1 over all windows
            self.classes_ = self.pipe_.classes_
            if self.aggregate in ("max", "lse") and len(self.classes_) < _MIL_MIN_CLASSES:
                # binary LDA.decision_function is 1-D (one margin, not per-class) — the score-pool has no
                # per-class axis to reduce over. concat/mean stay valid; MIL pooling needs >=3 classes.
                raise ValueError(f"aggregate={self.aggregate!r} needs >=3 classes (got {len(self.classes_)}); "
                                 "use 'concat' or 'mean' for binary")
        return self

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        Fw = self._window_feats(np.asarray(X))                           # [n, W, D]
        n, W, D = Fw.shape
        if self.aggregate == "concat":
            return self.pipe_.predict_proba(Fw.reshape(n, -1))
        if self.aggregate == "mean":                                     # average per-window probabilities
            p = self.pipe_.predict_proba(Fw.reshape(n * W, D)).reshape(n, W, -1)
            return p.mean(axis=1)
        # max / lse: pool the per-window class SCORES (LDA decision fn), then softmax to probabilities. MIL —
        # the block score for a class is its strongest (max) / soft-strongest (lse) sub-window, not the mean.
        s = self.pipe_.decision_function(Fw.reshape(n * W, D))
        s = s.reshape(n, W, -1) if s.ndim == _MULTICLASS_SCORE_NDIM else s.reshape(n, W, 1)   # [n, W, C]
        pooled = s.max(axis=1) if self.aggregate == "max" else _logsumexp(s, axis=1)
        return self._softmax(pooled)


def _logsumexp(z: np.ndarray, axis: int) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)
