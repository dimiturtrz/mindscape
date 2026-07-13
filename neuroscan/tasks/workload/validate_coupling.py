"""Coupling-extraction ground-truth check (bd uqw) — runnable diagnostic, not just a test.

Generate paired (HbO, HbR) from a KNOWN neural drive via the independent physics-forward synthetic
(`core.data.fnirs.synthetic`, a double-gamma HRF distinct from our single-gamma estimator), then confirm our
shipped operators recover the ground truth: CBSI (`chromophore.cbsi_neural`) separates the neural response from
the common-mode systemic, and `coupling.estimate_coupling` recovers the physiological lag + the positive sign.
Non-circular (forward shape != estimator shape). Run it to re-check the operators after any change:

    python -m neuroscan.tasks.workload.validate_coupling
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.signal import fftconvolve

from core.data.fnirs.synthetic import Synthetic, SynthConfig
from core.features.fnirs.chromophore import Chromophore
from core.features.fusion.coupling import Coupling

logger = logging.getLogger(__name__)

_FS = 5.0
_MIN_CORR = 0.9                 # CBSI must track the true neural response this well
_LAG_WINDOW = (2.0, 9.0)        # recovered lag must sit in this physiological window (around the 6 s HRF peak)


def run(n_seeds: int = 5) -> dict:
    """Recover CBSI-vs-neural correlation + coupling lag/sign across seeds against the synthetic ground truth."""
    cfg = SynthConfig()
    hrf = Synthetic.double_gamma_hrf(_FS, cfg)
    com = float((np.arange(len(hrf)) * hrf).sum() / hrf.sum() / _FS)
    corrs, lags, signs = [], [], []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        drive = np.zeros((1, 2000))
        for s in rng.integers(0, 1900, 25):
            drive[0, s:s + rng.integers(20, 60)] += 1.0
        hbo, hbr = Synthetic.synthesize_paired(drive, _FS, cfg, seed=seed + 100)
        cbsi = Chromophore.cbsi_neural(hbo[:, None, :], hbr[:, None, :])[:, 0, :]
        true_resp = fftconvolve(drive, hrf[None, :], axes=1)[:, :drive.shape[1]]
        corrs.append(float(np.corrcoef(cbsi[0], true_resp[0])[0, 1]))
        lag, _decay, beta = Coupling.estimate_coupling(drive, cbsi, _FS)
        lags.append(lag)
        signs.append(beta > 0)
    return {"hrf_com_s": com, "cbsi_corr_mean": float(np.mean(corrs)),
            "lag_mean_s": float(np.mean(lags)), "lag_std_s": float(np.std(lags)),
            "sign_positive_frac": float(np.mean(signs))}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    r = run()
    logger.info("coupling ground-truth validation (independent double-gamma forward):")
    logger.info(f"  CBSI vs true neural   : corr {r['cbsi_corr_mean']:.3f}  (separates neural from systemic)")
    logger.info(f"  estimate_coupling lag : {r['lag_mean_s']:.2f} ± {r['lag_std_s']:.2f} s "
          f"(HRF center-of-mass {r['hrf_com_s']:.2f} s)")
    logger.info(f"  coupling sign positive: {r['sign_positive_frac']*100:.0f}% of seeds")
    ok = (r["cbsi_corr_mean"] > _MIN_CORR and _LAG_WINDOW[0] <= r["lag_mean_s"] <= _LAG_WINDOW[1]
          and r["sign_positive_frac"] == 1.0)
    logger.info(f"  -> operators {'RECOVER ground truth' if ok else 'FAIL to recover ground truth'}")


if __name__ == "__main__":
    main()
