"""Decoder method registry — one place that maps a method name to its (fit_fn, score_fn) pair.

Unifies the handcrafted baseline (CSP+LDA, plain functions) and the braindecode decoders (built via
decoders.make) behind one lookup, so entrypoints (tasks (run, workload/*, motor_imagery/*)) don't each
re-implement the csp-vs-net branch. The harness contract is the same for all: `fit(X,y)->clf`,
`score(clf,X)->probs[n,C]`.
"""
from __future__ import annotations


def method_names() -> list[str]:
    from neuroscan.models.decoders import MODELS
    return ["csp_lda", "riemann", "riemann_acm", "riemann_mdm", "riemann_fgmdm", "fbcsp",
            "fnirs_lda", "fnirs_windowed", "fnirs_glm", "eeg_bandpower", *sorted(MODELS)]


# baselines that need the epoch sample rate: filter-designers (band-power, FBCSP) to build their filters,
# the windowed fNIRS decoder (sub-window seconds->samples), and GLM-β (HRF/task regressor timing).
_FS_METHODS = {"eeg_bandpower", "fbcsp", "fnirs_windowed", "fnirs_glm"}


def _proba(clf, X):
    """The single scorer for every Decoder — classical baseline or braindecode net both expose it."""
    return clf.predict_proba(X)


def _baseline_classes() -> dict:
    """name -> Baseline class (lazy import so pyriemann/mne load only when a baseline is actually used)."""
    from baselines.eeg.bandpower import EegBandpower
    from baselines.eeg.csp_lda import CspLda
    from baselines.eeg.fbcsp import Fbcsp
    from baselines.eeg.riemann import Acm, Fgmdm, Mdm, TangentSpace
    from baselines.fnirs.features import FnirsLda
    from baselines.fnirs.glm import GlmBeta
    from baselines.fnirs.windowed import WindowedFnirs
    return {"csp_lda": CspLda, "riemann": TangentSpace, "riemann_acm": Acm, "riemann_mdm": Mdm,
            "riemann_fgmdm": Fgmdm, "fbcsp": Fbcsp, "fnirs_lda": FnirsLda, "fnirs_windowed": WindowedFnirs,
            "fnirs_glm": GlmBeta, "eeg_bandpower": EegBandpower}


def get_method(name: str, fs: float | None = None):
    """Return (fit_fn, score_fn) for a method name. Every method is a Decoder (fit -> self, predict_proba),
    so the scorer is always `_proba`; only the builder differs — a fresh baseline object per fold, or a
    braindecode net built with its per-model cfg (decoders.make). `fs` (the epoch sample rate) is passed to
    the filter-designing baselines (band-power, FBCSP); ignored by the rest."""
    classes = _baseline_classes()
    if name in classes:
        cls = classes[name]
        kw = {"fs": fs} if (name in _FS_METHODS and fs is not None) else {}
        return (lambda X, y: cls(**kw).fit(X, y), _proba)
    from neuroscan.models.decoders import make
    fit, _ = make(name)                      # net builds its own cfg; its scorer is predict_proba too
    return (fit, _proba)
