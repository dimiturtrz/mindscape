"""FBCSP smoke: a band-specific spatial-covariance signal must decode through the filter bank + LDA.

Builds trials whose CLASS lives in the channel covariance of ONE sub-band (a distinct mixing applied to
band-limited noise), the exact structure filter-bank CSP is built to isolate — and checks the fit/predict
contract holds and the signal decodes above chance.
"""
import numpy as np

from baselines.eeg.fbcsp import Fbcsp


def _band_cov_dataset(n_per_class=40, n_ch=6, n_t=256, fs=128.0, seed=0):
    """Class encoded in the spatial mixing of a ~10-14 Hz component riding on broadband noise."""
    from mne.filter import filter_data
    rng = np.random.default_rng(seed)
    A0, A1 = rng.normal(size=(n_ch, n_ch)), rng.normal(size=(n_ch, n_ch))
    X, y = [], []
    for cls, A in ((0, A0), (1, A1)):
        for _ in range(n_per_class):
            base = rng.normal(size=(n_ch, n_t))
            band = filter_data(rng.normal(size=(n_ch, n_t)), fs, 10, 14, verbose=False)
            X.append(base + 3.0 * (A @ band))          # class-specific mixing inside one band
            y.append(cls)
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def test_fbcsp_decodes_band_covariance_signal():
    X, y = _band_cov_dataset(seed=1)
    clf = Fbcsp(fs=128.0, fmin=4.0, fmax=40.0, band_width=4.0, n_components=4, k_features=8).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(1), 1.0, atol=1e-5)
    assert (proba.argmax(1) == y).mean() > 0.7


def test_fbcsp_selects_at_most_k_features():
    X, y = _band_cov_dataset(n_per_class=20, seed=2)
    clf = Fbcsp(fs=128.0, k_features=5).fit(X, y)
    assert len(clf.sel_) == 5                          # MI keeps exactly k when the bank has more than k
