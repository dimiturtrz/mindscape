"""Cross-subject re-centering helpers: per-domain alignment moves each domain's mean to the identity,
and the covariance step grows correctly under time-delay augmentation. Synthetic — no dataset needed."""
import numpy as np

from neuroscan.experiments import align


def _domain(n, d, t, shift, rng):
    """n covariance-bearing trials whose cloud is displaced by an SPD `shift`."""
    return np.stack([shift @ (B @ B.T / t) @ shift for B in rng.normal(size=(n, d, t))])


def test_recenter_by_group_centers_each_domain_to_identity():
    from pyriemann.utils.mean import mean_riemann
    rng = np.random.default_rng(0)
    d = 4
    sa = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))   # domain A's location
    sb = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))   # domain B's, elsewhere
    Ca, Cb = _domain(25, d, 12, sa, rng), _domain(25, d, 12, sb, rng)
    C = np.concatenate([Ca, Cb])
    groups = np.array(["a"] * 25 + ["b"] * 25)

    Crc = align._recenter_by_group(C, groups)
    # each domain's Riemannian mean must land on I after its own re-centering
    assert np.allclose(mean_riemann(Crc[groups == "a"]), np.eye(d), atol=1e-4)
    assert np.allclose(mean_riemann(Crc[groups == "b"]), np.eye(d), atol=1e-4)


def test_covariances_augment_grows_channels_by_order():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 5, 200))                       # 6 trials, 5 ch, 200 samples
    plain = align._covariances(X, augment=False, order=4, lag=8)
    aug = align._covariances(X, augment=True, order=4, lag=8)
    assert plain.shape == (6, 5, 5)
    assert aug.shape == (6, 20, 20)                        # ch * order = 5 * 4
