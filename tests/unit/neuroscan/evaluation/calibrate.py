"""Calibrate runner aggregation — per-subject rows -> transfer-ratio summary + verdict."""
import numpy as np

from neuroscan.evaluation import calibrate


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
