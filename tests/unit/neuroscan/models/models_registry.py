"""Method registry — unified lookup for baseline + braindecode decoders."""
import numpy as np
import pytest

from baselines.eeg.fbcsp import FbcspConfig
from baselines.fnirs.windowed import WindowedConfig
from neuroscan import models


def test_get_method_returns_fit_score_callables():
    for name in ("csp_lda", "riemann", "eegnet"):
        fit, score = models.Methods.get_method(name)
        assert callable(fit) and callable(score)


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        models.Methods.get_method("nope")


def test_fs_config_methods_route_fs_into_a_config_object():
    # fbcsp / fnirs_windowed take fs as a FIELD on their config, not an fs= ctor kwarg
    assert models._FS_CONFIG == {"fbcsp": FbcspConfig, "fnirs_windowed": WindowedConfig}
    assert FbcspConfig(fs=200.0).fs == 200.0
    assert WindowedConfig(fs=10.0).fs == 10.0
    # the fs-config branch (builds the config) runs at get_method time
    for name in ("fbcsp", "fnirs_windowed"):
        fit, score = models.Methods.get_method(name, fs=125.0)
        assert callable(fit) and score is models.Methods._proba


def test_fs_kwarg_methods_and_no_fs_paths_return_proba_scorer():
    # bandpower takes fs=; csp_lda ignores fs entirely -> all share the single _proba scorer
    assert models.Methods.get_method("eeg_bandpower", fs=250.0)[1] is models.Methods._proba
    assert models.Methods.get_method("csp_lda", fs=None)[1] is models.Methods._proba


class _FakeClf:
    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


def test_proba_scorer_delegates_to_predict_proba():
    probs = models.Methods._proba(_FakeClf(), np.zeros((3, 4, 5)))
    assert probs.shape == (3, 2)
