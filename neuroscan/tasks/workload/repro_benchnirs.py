"""Reproduce the BenchNIRS 'generalised' (cross-subject) LDA baseline on the Shin n-back — the leakage-free,
matched reference our fnirs_lda is compared against.

BenchNIRS (Benerradi et al. 2023, Front. Neuroergonomics 4:994969) is the field's rigorous fNIRS-ML
benchmark; its whole point is that proper cross-subject evaluation gives near-chance numbers, exposing
that many published fNIRS accuracies are inflated by improper (personalised / within-session) validation.
Its loader reads the SAME VP*-NIRS/cnt_nback.mat files we already have, so we run its exact pipeline on
our data instead of reinventing it. Paper number: LDA 38.9% (3-class, chance 33.3%).

A reproduction entrypoint (alongside reproduce_atcnet) — BenchNIRS is an optional dependency, not a
pipeline dep:
    uv pip install benchnirs            # (isolated venv recommended; it pins its own mne)
    python -m neuroscan.tasks.workload.repro_benchnirs --data <dir with VP001-NIRS/…>

Expected: MEAN ≈ 0.39 (we measured 0.392 on our copy — matches the paper's 0.389). Compare against our
`fnirs_lda` under the matched regime: `run_fnirs --exp nback_fnirs_cross_kfold`.
"""
import argparse
import logging

import numpy as np

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    from benchnirs.learn import machine_learn  # noqa: PLC0415
    from benchnirs.load import load_dataset  # noqa: PLC0415
    from benchnirs.process import extract_features, process_epochs  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dir holding VP001-NIRS/cnt_nback.mat …")
    args = ap.parse_args()

    epochs = load_dataset("shin_2018_nb", args.data, bandpass=[0.01, 0.5],
                          baseline=(-2, 0), roi_sides=True, tddr=True)         # their exact preprocessing
    nirs, labels, groups = process_epochs(epochs[["0-back", "2-back", "3-back"]], tmax=9.9)
    feats = extract_features(nirs, ["mean", "std", "slope"])                   # their features (ROI-averaged)
    lda, _, _ = machine_learn("lda", feats, labels, groups, output_folder="./out_benchnirs_lda",
                              random_state=42)                                 # 5-fold GroupKFold, plain LDA
    logger.info(f"BenchNIRS LDA per-fold: {np.round(lda, 4)}")
    logger.info(f"MEAN {np.mean(lda):.4f} (sd {np.std(lda):.4f}) | chance 0.333 | paper 0.389")


if __name__ == "__main__":
    main()
