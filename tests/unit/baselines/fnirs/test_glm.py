"""GLM-β decoder — the design matrix layout, that β recovers the HRF-template amplitude (so class encoded in
response strength is separable), and the harness contract."""
import numpy as np

from baselines.fnirs.glm import GlmBeta, _canonical_hrf


def test_canonical_hrf_shape_and_peak():
    h = _canonical_hrf(fs=10.0)
    assert h.ndim == 1 and abs(h).max() == 1.0                      # peak-normalized
    assert h.argmax() / 10.0 < 8.0                                  # peaks within ~8 s (canonical ~5-6 s)


def test_design_columns_features_then_drift():
    clf = GlmBeta(fs=10.0, derivatives=True, drift_order=1)
    D, n_feat = clf._design(T=220)
    assert D.shape == (220, 3 + 2)                                  # HRF + 2 derivs (features) + const + linear (drift)
    assert n_feat == 3                                              # only the HRF/deriv βs are features
    no_d = GlmBeta(derivatives=False, drift_order=0)._design(220)
    assert no_d == (no_d[0], 1) or no_d[1] == 1                     # HRF-only -> 1 feature col


def test_beta_recovers_amplitude_and_separates():
    fs, T, ch = 10.0, 220, 4
    clf = GlmBeta(fs=fs, derivatives=False, drift_order=1)
    D, _ = clf._design(T)
    reg = D[:, 0]                                                   # the HRF regressor
    rng = np.random.default_rng(0)
    X, y = [], []
    for c, amp in {0: 0.5, 1: 1.5, 2: 3.0}.items():                # class = response amplitude
        for _ in range(40):
            sig = amp * reg[None, :] + rng.standard_normal((ch, T)) * 0.1
            X.append(sig); y.append(c)
    X = np.asarray(X); y = np.asarray(y)
    feats = clf._features(X)                                        # [n, ch*1]
    assert feats.shape == (len(y), ch)
    # the recovered β tracks the injected amplitude (monotone across classes)
    per_class = [feats[y == c].mean() for c in (0, 1, 2)]
    assert per_class[0] < per_class[1] < per_class[2]
    p = clf.fit(X, y).predict_proba(X)
    assert p.shape == (len(y), 3) and np.allclose(p.sum(1), 1.0, atol=1e-6)
    assert (p.argmax(1) == y).mean() > 0.9                          # amplitude-coded classes are separable
