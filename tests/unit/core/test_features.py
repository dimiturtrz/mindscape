"""core/features covariance-module transforms — the pure signal→feature ops (time-delay embedding, dispersion
scaling). Band-power / amplitude are exercised via the baselines that call them (test_bandpower, test_fnirs)."""
import numpy as np

from core.features import scale_to_identity, time_delay_embed


def test_time_delay_embed_grows_channels_by_order():
    X = np.random.default_rng(1).normal(size=(6, 5, 200))              # 6 trials, 5 ch, 200 samples
    aug = time_delay_embed(X, order=4, lag=8)
    assert aug.shape == (6, 5 * 4, 200 - 3 * 8)                        # ch*order rows, t-(order-1)*lag cols


def test_time_delay_embed_rejects_too_long_embedding():
    import pytest
    with pytest.raises(ValueError):
        time_delay_embed(np.zeros((2, 3, 10)), order=4, lag=8)        # 4*8 > 10


def test_scale_to_identity_normalizes_dispersion():
    """RPA step 2: after scaling, the mean squared Riemannian distance to I equals the target dispersion,
    regardless of the cloud's original spread."""
    from pyriemann.utils.distance import distance_riemann
    rng = np.random.default_rng(2)
    d = 4
    C = np.stack([3 * np.eye(d) @ (B @ B.T / 15) @ (3 * np.eye(d)) for B in rng.normal(size=(30, d, 15))])
    Cs = scale_to_identity(C, target_disp=1.0)
    disp = np.mean([distance_riemann(c, np.eye(d)) ** 2 for c in Cs])
    assert abs(disp - 1.0) < 1e-6
