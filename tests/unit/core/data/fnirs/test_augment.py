"""Domain-randomization augmentation (bd jdh) — perturbs nuisance axes, preserves the neural CBSI contrast."""
import numpy as np

from core.data.fnirs.augment import Augment, AugConfig
from core.data.fnirs.synthetic import Synthetic, SynthConfig
from core.features.fnirs.chromophore import Chromophore


def _paired(n=4, ch=6, t=400):
    drive = (np.random.default_rng(0).standard_normal((n * ch, t)) > 1.0).astype(float)
    hbo, hbr = Synthetic.synthesize_paired(drive, 5.0, SynthConfig(noise_std=0.02), seed=0)
    return hbo.reshape(n, ch, t), hbr.reshape(n, ch, t)


def test_shape_preserved_and_not_identity():
    hbo, hbr = _paired()
    ao, ar = Augment.domain_randomize(hbo, hbr, 5.0, seed=1)
    assert ao.shape == hbo.shape and ao.dtype == np.float32
    assert not np.allclose(ao, hbo)                              # it actually augments


def test_cbsi_neural_contrast_survives():
    """With timing shift off, the augmented CBSI still tracks the original — gain scales both chromophores
    equally and the added systemic is common-mode, so CBSI cancels the injected nuisance."""
    hbo, hbr = _paired()
    ao, ar = Augment.domain_randomize(hbo, hbr, 5.0, AugConfig(max_shift_s=0.0), seed=2)
    base = Chromophore.cbsi_neural(hbo, hbr)
    aug = Chromophore.cbsi_neural(ao, ar)
    corr = np.mean([np.corrcoef(base[i, 0], aug[i, 0])[0, 1] for i in range(hbo.shape[0])])
    assert corr > 0.8                                            # neural contrast preserved through the aug
