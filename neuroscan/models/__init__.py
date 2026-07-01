"""Decoder method registry — one place that maps a method name to its (fit_fn, score_fn) pair.

Unifies the handcrafted baseline (CSP+LDA, plain functions) and the braindecode decoders (built via
decoders.make) behind one lookup, so entrypoints (experiments/run, calibrate, quantize) don't each
re-implement the csp-vs-net branch. The harness contract is the same for all: `fit(X,y)->clf`,
`score(clf,X)->probs[n,C]`.
"""
from __future__ import annotations


def method_names() -> list[str]:
    from neuroscan.models.decoders import MODELS
    return ["csp_lda", "riemann", "riemann_acm", "fnirs_lda", *sorted(MODELS)]


def _adapt(cls, **hp):
    """Adapt a Baseline class to the harness (fit_fn, score_fn) contract — a fresh instance is built and
    fitted per fold, and the fitted object scores itself."""
    return (lambda X, y: cls(**hp).fit(X, y), lambda clf, X: clf.score(X))


def get_method(name: str):
    """Return (fit_fn, score_fn) for a method name. Baselines are Baseline classes adapted to the
    contract; braindecode decoders come from decoders.make."""
    if name == "csp_lda":
        from baselines.csp_lda import CspLda
        return _adapt(CspLda)
    if name == "riemann":
        from baselines.riemann import TangentSpace
        return _adapt(TangentSpace)
    if name == "riemann_acm":
        from baselines.riemann import Acm
        return _adapt(Acm)
    if name == "fnirs_lda":
        from baselines.fnirs_features import FnirsLda
        return _adapt(FnirsLda)
    from neuroscan.models.decoders import make
    return make(name)
