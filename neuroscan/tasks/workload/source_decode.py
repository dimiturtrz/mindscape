"""Source-space decode probe (bd 728) — does projecting EEG to cortical parcels beat sensor space cross-subject?

Wires the 728 source operator into a real decode. Workload EEG (Shin n-back), per-subject **sensor** covariance
vs **source-parcel** covariance (`source.to_parcels`: fsaverage forward + dSPM inverse -> 68 Desikan-Killiany
parcels), run through the exact winning cross-subject method — per-subject re-centered tangent + LR
(`transfer.zero_shot_predict`). Beat the ~0.58 EEG-sensor reference = the cortical projection carries
cross-subject-transferable structure the volume-conducted sensor mixture blurs; ≤ it = the sensor covariance
already captures it (a fair null — source localization on a template head adds regularization, not information).

    python -m neuroscan.tasks.workload.source_decode
"""
from __future__ import annotations

import logging

import numpy as np
from pyriemann.estimation import Covariances
from sklearn.model_selection import StratifiedGroupKFold

from baselines.eeg import transfer
from core.data import store
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.eeg.base import EpochCfg
from core.features.eeg.source import to_parcels
from neuroscan.evaluation import metrics

logger = logging.getLogger(__name__)

_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_SEEDS, _K = [0], 5          # 1 seed default (escalate-on-signal, bd); k-fold = validity not rigor
_SENSOR_REF = 0.580        # EEG-sensor re-centered Riemann cross-subject reference (workload n-back)


def _cov(x: np.ndarray) -> np.ndarray:
    return Covariances("oas").transform(x.astype(np.float64))


def _build():
    """Per-subject sensor + source covariances for the workload EEG blocks."""
    meta = store.load("shin2017_nback_eeg", _CFG)
    ch = eegmod.adapter().channels()
    subs = sorted(meta["subject"].unique().to_list())
    c_sensor, c_source, ys, gs = [], [], [], []
    for s in subs:
        x, y = store.gather(meta.filter(meta["subject"] == s))
        c_sensor.append(_cov(x))
        c_source.append(_cov(to_parcels(x, ch, _CFG.resample)))       # [n,68,t] cortical parcels
        ys.append(y)
        gs.append(np.array([s] * len(y)))
        logger.info(f"  subject {s}: {len(y)} blocks -> sensor {c_sensor[-1].shape[1:]}, source {c_source[-1].shape[1:]}")
    return (np.concatenate(c_sensor), np.concatenate(c_source),
            np.concatenate(ys), np.concatenate(gs))


def _decode(c: np.ndarray, y: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    """Cross-subject re-centered tangent + LR, `_SEEDS` x `_K`-fold grouped by subject."""
    accs = []
    for seed in _SEEDS:
        for tr, te in StratifiedGroupKFold(_K, shuffle=True, random_state=seed).split(c, y, g):
            proba = transfer.zero_shot_predict(transfer.Domain(c[tr], y[tr], g[tr]),
                                               transfer.Domain(c[te], groups=g[te]), scale=False)
            accs.append(metrics.accuracy(y[te], proba.argmax(1)))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for lib_name in ("mne", "moabb", "braindecode"):
        logging.getLogger(lib_name).setLevel(logging.WARNING)
    c_sensor, c_source, y, g = _build()
    logger.info(f"\n{c_sensor.shape[0]} blocks · {len(set(g.tolist()))} subj · chance {1 / (y.max() + 1):.3f} "
          f"· {len(_SEEDS)}x{_K}-fold re-centered tangent")
    a_sen, s_sen = _decode(c_sensor, y, g)
    a_src, s_src = _decode(c_source, y, g)
    logger.info(f"  sensor  (28 ch)      acc {a_sen:.3f} ± {s_sen:.3f}")
    logger.info(f"  source  (68 parcels) acc {a_src:.3f} ± {s_src:.3f}")
    verdict = "source CASHES structure" if a_src > a_sen + 0.01 else "fair null (source adds no decodable info)"
    logger.info(f"  Δ source − sensor: {a_src - a_sen:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
