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
    T = calibrate.TemperatureScaler().fit(logits, y).T
    assert T > 1.0                           # overconfident -> T>1 softens


def test_temperature_lowers_ece_and_keeps_argmax():
    logits, y = _overconfident()
    T = calibrate.TemperatureScaler().fit(logits, y).T
    assert calibrate.TemperatureScaler(T).ece(logits, y) < calibrate.TemperatureScaler(1.0).ece(logits, y)
    # argmax (accuracy) unchanged by temperature
    assert np.array_equal((logits / T).argmax(1), logits.argmax(1))


def _row(val_uncal, val_temp, test_uncal, test_temp, t=2.0):
    return {"T": t, "val_ece_uncal": val_uncal, "val_ece_temp": val_temp,
            "test_ece_uncal": test_uncal, "test_ece_temp": test_temp}


def test_summarize_aggregates_and_computes_transfer_ratio():
    rows = [_row(0.20, 0.05, 0.30, 0.24), _row(0.20, 0.05, 0.30, 0.24)]   # val fix 0.15, test fix 0.06
    summary, val_fix, test_fix = calibrate.Calibrate._summarize(rows, "atcnet")
    assert summary["method"] == "atcnet" and summary["n"] == 2
    assert np.isclose(val_fix, 0.15) and np.isclose(test_fix, 0.06)
    assert np.isclose(summary["transfer_ratio"], 0.4)                     # 0.06 / 0.15


def test_summarize_transfer_ratio_none_when_val_already_calibrated():
    rows = [_row(0.10, 0.10, 0.20, 0.15)]                                 # val fix ~0 -> ratio undefined
    summary, val_fix, _test_fix = calibrate.Calibrate._summarize(rows, "m")
    assert val_fix <= 1e-6
    assert summary["transfer_ratio"] is None


def _verdict(val_uncal, val_temp, test_uncal, test_temp):
    rows = [_row(val_uncal, val_temp, test_uncal, test_temp)]
    summary, val_fix, test_fix = calibrate.Calibrate._summarize(rows, "m")
    calibrate.Calibrate._report(summary, "m", val_fix, test_fix)
    return summary["verdict"]


def test_report_verdict_covers_each_transfer_regime():
    assert "nothing to transfer" in _verdict(0.10, 0.10, 0.20, 0.15)      # ratio None
    assert "LIMITED" in _verdict(0.20, 0.00, 0.30, 0.27)                  # ratio 0.10 < 0.5
    assert "partially" in _verdict(0.20, 0.00, 0.30, 0.16)               # ratio 0.70 in [0.5, 1.2)
    assert "transfers well" in _verdict(0.20, 0.00, 0.30, 0.00)          # ratio 1.5 >= 1.2
