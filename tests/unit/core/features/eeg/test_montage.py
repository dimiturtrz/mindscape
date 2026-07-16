"""EEG scalp geometry (`eeg_positions` + `to_unit_disk`) — pure lookup/geometry, no disk.

Reads the in-memory MNE standard 10-05 montage and normalizes to the unit head disk: known channels land at
finite positions inside the disk, unknown channels become NaN rows.
"""
import numpy as np

from core.features.eeg.montage import EegMontage


def test_eeg_positions_shape_and_unit_disk():
    ch = ["Cz", "Fz", "Pz", "Oz", "C3", "C4"]
    pos = EegMontage.eeg_positions(ch)
    assert pos.shape == (len(ch), 2)
    r = np.hypot(pos[:, 0], pos[:, 1])
    assert np.all(r <= 1.0 + 1e-6)                          # normalized into the unit disk
    assert np.isfinite(pos).all()


def test_unknown_channel_is_nan_row():
    pos = EegMontage.eeg_positions(["Cz", "NOT_A_CHANNEL"])
    assert np.isnan(pos[1]).all()                           # missing channel -> NaN, not a fabricated position


def testto_unit_disk_centers_and_scales():
    raw = np.array([[10.0, 10.0], [10.0, 12.0], [12.0, 10.0], [8.0, 8.0]])
    out = EegMontage.to_unit_disk(raw)
    assert np.allclose(out.mean(0), 0.0, atol=1e-6)        # centred on the mean
    assert np.hypot(out[:, 0], out[:, 1]).max() <= 1.0 + 1e-6


def test_channel_laplacian_is_valid_graph_laplacian():
    pos = EegMontage.eeg_positions(["Cz", "Fz", "Pz", "Oz", "C3", "C4"])
    lap = EegMontage.channel_laplacian(pos, sigma=0.3)
    assert lap.shape == (len(pos), len(pos))
    assert np.allclose(lap, lap.T, atol=1e-6)              # symmetric (undirected graph)
    assert np.allclose(lap.sum(1), 0.0, atol=1e-5)         # row sums zero: L = D − A, constant is the null vector
    eig = np.linalg.eigvalsh(lap)
    assert eig.min() >= -1e-5                              # positive-semidefinite (a real Laplacian)


def test_channel_laplacian_penalizes_neighbor_difference():
    pos = EegMontage.eeg_positions(["Cz", "Fz", "Pz", "Oz", "C3", "C4"])
    lap = EegMontage.channel_laplacian(pos, sigma=0.3)
    f_const = np.ones(len(pos))
    f_rough = np.arange(len(pos), dtype=float)
    assert abs(f_const @ lap @ f_const) < 1e-5            # smooth (constant) signal: zero penalty
    assert f_rough @ lap @ f_rough > 0.0                  # a signal varying across neighbours pays a positive cost


def test_channel_laplacian_narrower_sigma_fewer_edges():
    pos = EegMontage.eeg_positions(["Cz", "Fz", "Pz", "Oz", "C3", "C4"])
    deg_wide = np.diag(EegMontage.channel_laplacian(pos, sigma=0.5)).sum()
    deg_narrow = np.diag(EegMontage.channel_laplacian(pos, sigma=0.15)).sum()
    assert deg_narrow < deg_wide                          # tighter RBF -> weaker/fewer edges -> lower total degree
