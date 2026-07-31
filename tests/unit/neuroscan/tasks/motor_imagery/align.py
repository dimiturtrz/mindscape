"""The align runner's own logic — covariance estimation with time-delay augmentation. The transfer *methods*
it delegates to are tested in tests/unit/baselines/eeg/test_transfer.py; the covariance/scale feature ops in
tests/unit/core/test_features.py."""
import numpy as np

from neuroscan.tasks.motor_imagery import align


def test_covariances_augment_grows_channels_by_order():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 5, 200))                       # 6 trials, 5 ch, 200 samples
    plain = align.Align._covariances(X, augment=False, order=4, lag=8)
    aug = align.Align._covariances(X, augment=True, order=4, lag=8)
    assert plain.shape == (6, 5, 5)
    assert aug.shape == (6, 20, 20)                        # ch * order = 5 * 4
