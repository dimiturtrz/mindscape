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

from core.data import store
from core.data.eeg import shin2017_nback_eeg as eegmod
from core.data.eeg.base import EpochCfg
from core.features.eeg.source import Source
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.workload.riemann import Riemann

logger = logging.getLogger(__name__)

_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_SEEDS, _K = [0], 5          # 1 seed default (escalate-on-signal, bd); k-fold = validity not rigor
_SENSOR_REF = 0.580        # EEG-sensor re-centered Riemann cross-subject reference (workload n-back)


class SourceDecode:
    """Source-space decode probe helpers (bd 728) — the free helpers folded in as staticmethods."""

    @staticmethod
    def _build():
        """Per-subject sensor + source covariances for the workload EEG blocks."""
        meta = store.Store.load("shin2017_nback_eeg", _CFG)
        ch = eegmod.Shin2017NbackEegAdapter.adapter().channels()
        subs = sorted(meta["subject"].unique().to_list())
        c_sensor, c_source, ys, gs = [], [], [], []
        for s in subs:
            x, y = store.Store.gather(meta.filter(meta["subject"] == s))
            c_sensor.append(Riemann.cov(x))
            c_source.append(Riemann.cov(Source.to_parcels(x, ch, _CFG.resample)))   # [n,68,t] cortical parcels
            ys.append(y)
            gs.append(np.array([s] * len(y)))
            logger.info(f"  subject {s}: {len(y)} blocks -> "
                        f"sensor {c_sensor[-1].shape[1:]}, source {c_source[-1].shape[1:]}")
        return (np.concatenate(c_sensor), np.concatenate(c_source),
                np.concatenate(ys), np.concatenate(gs))


def main():
    Cli.setup_logging()
    c_sensor, c_source, y, g = SourceDecode._build()
    logger.info(f"\n{c_sensor.shape[0]} blocks · {len(set(g.tolist()))} subj · chance {1 / (y.max() + 1):.3f} "
          f"· {len(_SEEDS)}x{_K}-fold re-centered tangent")
    a_sen, s_sen = Riemann.cross_subject_decode(c_sensor, y, g, _SEEDS, _K)
    a_src, s_src = Riemann.cross_subject_decode(c_source, y, g, _SEEDS, _K)
    logger.info(f"  sensor  (28 ch)      acc {a_sen:.3f} ± {s_sen:.3f}")
    logger.info(f"  source  (68 parcels) acc {a_src:.3f} ± {s_src:.3f}")
    verdict = "source CASHES structure" if a_src > a_sen + 0.01 else "fair null (source adds no decodable info)"
    logger.info(f"  Δ source − sensor: {a_src - a_sen:+.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
