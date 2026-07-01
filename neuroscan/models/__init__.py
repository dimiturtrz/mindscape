"""Decoder method registry — one place that maps a method name to its (fit_fn, score_fn) pair.

Unifies the handcrafted baseline (CSP+LDA, plain functions) and the braindecode decoders (built via
decoders.make) behind one lookup, so entrypoints (tasks (run, workload/*, motor_imagery/*)) don't each
re-implement the csp-vs-net branch. The harness contract is the same for all: `fit(X,y)->clf`,
`score(clf,X)->probs[n,C]`.
"""
from __future__ import annotations


def method_names() -> list[str]:
    from neuroscan.models.decoders import MODELS
    return ["csp_lda", "riemann", "riemann_acm", "fnirs_lda", "eeg_bandpower", *sorted(MODELS)]


def _proba(clf, X):
    """The single scorer for every Decoder — classical baseline or braindecode net both expose it."""
    return clf.predict_proba(X)


def _baseline_classes() -> dict:
    """name -> Baseline class (lazy import so pyriemann/mne load only when a baseline is actually used)."""
    from baselines.csp_lda import CspLda
    from baselines.eeg_bandpower import EegBandpower
    from baselines.fnirs_features import FnirsLda
    from baselines.riemann import Acm, TangentSpace
    return {"csp_lda": CspLda, "riemann": TangentSpace, "riemann_acm": Acm, "fnirs_lda": FnirsLda,
            "eeg_bandpower": EegBandpower}


def get_method(name: str):
    """Return (fit_fn, score_fn) for a method name. Every method is a Decoder (fit -> self, predict_proba),
    so the scorer is always `_proba`; only the builder differs — a fresh baseline object per fold, or a
    braindecode net built with its per-model cfg (decoders.make)."""
    classes = _baseline_classes()
    if name in classes:
        cls = classes[name]
        return (lambda X, y: cls().fit(X, y), _proba)
    from neuroscan.models.decoders import make
    fit, _ = make(name)                      # net builds its own cfg; its scorer is predict_proba too
    return (fit, _proba)
