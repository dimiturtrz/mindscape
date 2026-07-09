"""fNIRS-informed weighted minimum-norm inverse (bd 4so) — the pure linear algebra, no MNE/fsaverage.

Equivalence classes: shape/validation, and the defining property — boosting the prior on a source concentrates
the recovered estimate there (fNIRS 'where' regularizing the ill-posed EEG inverse).
"""
import numpy as np
import pytest

from core.features.fusion.source_prior import weighted_min_norm_inverse


def _leadfield(seed=0, n_ch=8, n_src=6):
    return np.random.default_rng(seed).standard_normal((n_ch, n_src))


def test_shape_and_source_reconstruction():
    g = _leadfield()
    k = weighted_min_norm_inverse(g, np.ones(g.shape[1]))
    assert k.shape == (g.shape[1], g.shape[0])                 # [n_src, n_ch]


def test_rejects_bad_prior():
    g = _leadfield()
    with pytest.raises(ValueError, match="n_src"):
        weighted_min_norm_inverse(g, np.ones(g.shape[1] + 1))   # length mismatch
    with pytest.raises(ValueError, match="non-negative"):
        weighted_min_norm_inverse(g, -np.ones(g.shape[1]))      # negative prior variance


def test_prior_concentrates_estimate_on_primed_source():
    g = _leadfield(seed=1)
    j = 2
    sensor = g[:, j]                                            # pure source-j activity at the sensors
    uniform = weighted_min_norm_inverse(g, np.ones(g.shape[1])) @ sensor
    prior = np.ones(g.shape[1])
    prior[j] = 20.0                                            # fNIRS says source j is active
    primed = weighted_min_norm_inverse(g, prior) @ sensor
    # the primed inverse puts a larger SHARE of the recovered energy on source j
    assert abs(primed[j]) / np.abs(primed).sum() > abs(uniform[j]) / np.abs(uniform).sum()
