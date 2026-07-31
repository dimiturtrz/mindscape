"""core.data.recipe — the SupportsKey recipe→cache-key contract.

Both modality configs must satisfy it structurally, so the modality-agnostic Store can take the one contract
instead of casting one config through the other. Guards that neither EpochCfg (EEG) nor FnirsCfg (fNIRS)
drifts off the `key() -> str` surface Store relies on.
"""
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from core.data.recipe import SupportsKey


def test_key():
    """Both recipes satisfy SupportsKey — each `key()` returns a non-empty cache-key string."""
    for cfg in (EpochCfg(), FnirsCfg()):
        recipe: SupportsKey = cfg          # structural conformance to the contract Store takes
        assert isinstance(recipe.key(), str) and recipe.key()
