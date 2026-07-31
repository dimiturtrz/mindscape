"""Shared Riemannian cross-subject decode (`neuroscan.tasks.workload.riemann.Riemann`).

`cov` must build a valid SPD covariance per epoch: `[n, ch, t]` -> `[n, ch, ch]`, symmetric, positive-definite
(OAS shrinkage guarantees PD even when t < ch). `cross_subject_decode` is smoke-tested on a tiny 2-group
covariance-encoded set: it must return (mean, std) accuracies in [0, 1].
"""
import numpy as np
import pytest

pytest.importorskip("pyriemann")

from neuroscan.tasks.workload.riemann import Riemann   # noqa: E402


def test_cov_is_spd_and_right_shape():
    """OAS covariance per epoch: square [n, ch, ch], symmetric, and positive-definite (all eigenvalues > 0)."""
    rng = np.random.default_rng(0)
    n, ch, t = 5, 4, 128
    x = rng.standard_normal((n, ch, t))
    c = Riemann.cov(x)
    assert c.shape == (n, ch, ch)
    assert np.allclose(c, np.transpose(c, (0, 2, 1)), atol=1e-8)     # symmetric
    eig = np.linalg.eigvalsh(c)
    assert (eig > 0).all()                                          # positive-definite (OAS shrinkage)


def test_cov_shrinkage_keeps_spd_when_undersampled():
    """t < ch would make the raw sample covariance singular; OAS shrinkage must still yield an SPD matrix."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 6, 4))                              # 4 samples, 6 channels -> rank-deficient raw
    eig = np.linalg.eigvalsh(Riemann.cov(x))
    assert (eig > 0).all()


def _two_group_dataset(rng):
    """Covariance-encoded 2-class set across 4 subjects: each class has its own channel-mixing matrix."""
    a0, a1 = rng.standard_normal((4, 4)), rng.standard_normal((4, 4))
    x, y, g = [], [], []
    for subj in range(4):
        for cls, a in ((0, a0), (1, a1)):
            for _ in range(6):
                x.append(a @ rng.standard_normal((4, 96)))
                y.append(cls)
                g.append(subj)
    return np.asarray(x), np.asarray(y), np.asarray(g)


def test_cross_subject_decode_returns_accuracies_in_unit_range():
    """Tiny grouped-CV smoke: (mean, std) accuracy, both in [0, 1]."""
    rng = np.random.default_rng(2)
    x, y, g = _two_group_dataset(rng)
    c = Riemann.cov(x)
    mean, std = Riemann.cross_subject_decode(c, y, g, seeds=[0], k=2)
    assert 0.0 <= mean <= 1.0
    assert 0.0 <= std <= 1.0
