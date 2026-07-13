"""Brain-camera rasterization (`core.features.fusion.camera.BrainCamera`).

The pure-numpy viz branch. Contracts: `build_tensor` -> `[n, C=5, grid, grid, T]` finite; `fused_node_series`
-> EEG channel format `[n, n_e_finite, T]` dropping non-finite-position nodes; `coverage_map` bounded in [0,1]
and local (high where both modalities co-locate, low far away).
"""
import numpy as np

from core.features.fusion.camera import BrainCamera, PairedModalities
from core.features.fusion.series import SeriesConfig


def _paired(rng, n=3, n_e=8, n_f=6, drop_node=False):
    Xe = rng.standard_normal((n, n_e, 4000))             # EEG @100 Hz, 40 s
    Xf = rng.standard_normal((n, 2 * n_f, 320))          # HbO+HbR @10 Hz, 32 s
    pos_e = rng.uniform(-0.8, 0.8, (n_e, 2))
    pos_f = rng.uniform(-0.8, 0.8, (n_f, 2))
    if drop_node:
        pos_e[-1] = np.nan
    return PairedModalities(Xe, Xf, pos_e, pos_f)


def test_build_tensor_shape_and_finite():
    rng = np.random.default_rng(0)
    grid = 12
    X = BrainCamera.build_tensor(_paired(rng), grid=grid, series=SeriesConfig(fps=10.0, t_end=20.0))
    assert X.shape == (3, 5, grid, grid, 200)            # C = [theta, alpha, beta, CBSI, coverage]
    assert np.isfinite(X).all()


def test_fused_node_series_drops_nonfinite_nodes_and_is_finite():
    rng = np.random.default_rng(1)
    joint, coupling = BrainCamera.fused_node_series(_paired(rng, drop_node=True),
                                                    series=SeriesConfig(fps=10.0, t_end=20.0))
    assert joint.shape == (3, 7, 200)                    # 8 EEG nodes - 1 non-finite = 7
    assert np.isfinite(joint).all()
    assert set(coupling) == {"lag", "decay", "beta"}


def test_fnirs_false_isolates_eeg_strength_nonnegative():
    """`fnirs=False` returns the EEG band-strength envelope alone (no combiner) — an envelope, so >= 0."""
    rng = np.random.default_rng(2)
    strength, _ = BrainCamera.fused_node_series(_paired(rng), fnirs=False,
                                                series=SeriesConfig(fps=10.0, t_end=20.0))
    assert strength.shape == (3, 8, 200)
    assert (strength >= 0).all()


def test_coverage_map_is_bounded_and_local():
    """Coverage in [0,1]; ~1 where an EEG and an fNIRS sensor co-locate (grid centre), lower far from both."""
    pos_e = np.array([[0.0, 0.0], [0.6, 0.6]])
    pos_f = np.array([[0.0, 0.0], [-0.6, -0.6]])
    cov = BrainCamera.coverage_map(pos_e, pos_f, grid=16)
    assert cov.shape == (16, 16)
    assert cov.min() >= 0.0 and cov.max() <= 1.0
    assert cov[8, 8] > cov[0, 0]                         # centre (co-located) beats a far corner
