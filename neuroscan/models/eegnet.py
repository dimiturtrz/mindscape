"""Back-compat shim — EEGNet now lives behind the shared braindecode trainer (models/decoders.py).
Kept so `--method eegnet` and any direct imports resolve to the same (fit, score) contract."""
from __future__ import annotations

from neuroscan.models.decoders import make

fit, score = make("eegnet")
