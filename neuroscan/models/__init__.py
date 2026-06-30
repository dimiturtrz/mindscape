"""Decoder method registry — one place that maps a method name to its (fit_fn, score_fn) pair.

Unifies the handcrafted baseline (CSP+LDA, plain functions) and the braindecode decoders (built via
decoders.make) behind one lookup, so entrypoints (experiments/run, calibrate, quantize) don't each
re-implement the csp-vs-net branch. The harness contract is the same for all: `fit(X,y)->clf`,
`score(clf,X)->probs[n,C]`.
"""
from __future__ import annotations


def method_names() -> list[str]:
    from neuroscan.models.decoders import MODELS
    return ["csp_lda", "riemann", "riemann_acm", *sorted(MODELS)]


def get_method(name: str):
    """Return (fit_fn, score_fn) for a method name (baseline or braindecode decoder)."""
    if name == "csp_lda":
        from baselines import csp_lda
        return csp_lda.fit, csp_lda.score
    if name == "riemann":
        from baselines import riemann
        return riemann.fit, riemann.score
    if name == "riemann_acm":
        import functools

        from baselines import riemann
        return functools.partial(riemann.fit, method="acm"), riemann.score
    from neuroscan.models.decoders import make
    return make(name)
