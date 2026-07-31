"""CspLda — the canonical motor-imagery baseline: Common Spatial Patterns -> LDA on raw [n, ch, t].

Synthetic lateralized trials (class 0 = variance on channel A, class 1 = variance on channel B) — exactly
the spatial-covariance signal CSP reads. Checks the fit/predict_proba contract, the shape, and that a clean
covariance-structure signal decodes above chance. Plus the module-level back-compat shims.
"""
import numpy as np

from baselines.eeg import csp_lda
from baselines.eeg.csp_lda import CspLda

N_CH = 6
N_T = 200


def _lateralized(n_per_class=25, seed=0):
    """class 0 = high-variance oscillation on channel 0; class 1 = on channel 1 — a spatial-covariance
    contrast (mu/beta-ERD style), the CSP-native signal."""
    rng = np.random.default_rng(seed)
    t = np.arange(N_T) / 100.0
    X, y = [], []
    for cls in (0, 1):
        for _ in range(n_per_class):
            sig = 0.3 * rng.normal(size=(N_CH, N_T))
            sig[cls] += 2.0 * np.sin(2 * np.pi * 12.0 * t + rng.uniform(0, 6.28))
            X.append(sig)
            y.append(cls)
    return np.asarray(X, float), np.asarray(y)


def test_fit_returns_self_and_proba_contract():
    X, y = _lateralized(n_per_class=20)
    clf = CspLda(n_components=4)
    assert clf.fit(X, y) is clf                             # fit -> self (Baseline contract)
    p = clf.predict_proba(X)
    assert p.shape == (len(y), 2)
    assert np.allclose(p.sum(1), 1.0, atol=1e-6)


def test_decodes_lateralized_covariance_signal():
    Xtr, ytr = _lateralized(seed=1)
    Xte, yte = _lateralized(seed=2)
    clf = CspLda().fit(Xtr, ytr)
    acc = (clf.predict_proba(Xte).argmax(1) == yte).mean()
    assert acc > 0.8                                        # a clean spatial-covariance signal must decode


def test_module_shims_delegate_to_class():
    X, y = _lateralized(n_per_class=20)
    clf = csp_lda.fit(X, y, n_components=4)
    probs = csp_lda.score(clf, X)
    assert probs.shape == (len(y), 2)
