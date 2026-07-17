"""ViewData.prediction_report — the shared per-class LOSO report + score dict."""
import numpy as np

from neuroviz.viewdata import Decode, ViewData


def test_prediction_report_builds_per_class_rows_and_score():
    id2name = {0: "left", 1: "right"}
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 1, 1, 0])                        # 2/4 correct
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
    per, score = ViewData.prediction_report(id2name, Decode(y, pred, probs, 0.5, "demo"))

    assert set(per) == {"left", "right"}
    # first shown "left" trial (index 0) predicted left -> correct; first "right" (index 2) predicted right
    assert per["left"] == {"truth": "left", "pred": "left", "probs": [0.9, 0.1], "correct": True}
    assert per["right"] == {"truth": "right", "pred": "right", "probs": [0.2, 0.8], "correct": True}
    assert score == {"acc": 0.5, "chance": 0.5, "regime": "cross-subject (LOSO)", "decoder": "demo"}
