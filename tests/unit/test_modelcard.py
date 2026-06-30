"""Model card renders the honest sections from an aggregate dict."""
from neuroscan.evaluation import modelcard

_RES = {
    "method": "csp_lda", "regime": "within",
    "fold_mean": {"acc": 0.60, "kappa": 0.47, "ece": 0.14},
    "pooled": {"acc": 0.60, "kappa": 0.47, "ece": 0.14},
    "acc_spread": {"mean": 0.60, "std": 0.12, "min": 0.34, "max": 0.73},
    "per_fold": [
        {"fold": "1", "acc": 0.34, "kappa": 0.10, "ece": 0.20, "n": 288},
        {"fold": "2", "acc": 0.73, "kappa": 0.64, "ece": 0.10, "n": 288},
    ],
}


def test_card_has_headline_and_where_it_fails():
    md = modelcard.card(_RES, "bnci2014_001", "within")
    assert "# Model card" in md
    assert "Where it fails" in md
    assert "worst: subject 1" in md          # lowest-acc fold surfaced
    assert "vs reference" in md
    assert "| 1 | 0.340 |" in md             # per-subject table row
