"""Temperature scaling: T softens overconfident logits and lowers ECE (argmax unchanged)."""
import numpy as np

from neuroscan.evaluation import calibrate


def _overconfident():
    # logits with large magnitude but ~half wrong -> overconfident, miscalibrated
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, size=200)
    logits = np.full((200, 4), -3.0)
    logits[np.arange(200), y] = 6.0          # very confident on the "true" class
    flip = rng.random(200) < 0.4             # but flip 40% to a wrong class -> overconfident
    wrong = (y + 1) % 4
    logits[flip, y[flip]], logits[flip, wrong[flip]] = -3.0, 6.0
    return logits, y


def test_fit_temperature_softens_overconfidence():
    logits, y = _overconfident()
    T = calibrate.fit_temperature(logits, y)
    assert T > 1.0                           # overconfident -> T>1 softens


def test_temperature_lowers_ece_and_keeps_argmax():
    logits, y = _overconfident()
    T = calibrate.fit_temperature(logits, y)
    assert calibrate.ece_at(logits, y, T) < calibrate.ece_at(logits, y, 1.0)
    # argmax (accuracy) unchanged by temperature
    assert np.array_equal((logits / T).argmax(1), logits.argmax(1))
