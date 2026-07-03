"""fNIRS subset study — the scale-free readout math (effective-#-features and the knee) plus a tiny check
that the differentiable weighted head actually learns to up-weight the informative family."""
import numpy as np

from neuroscan.tasks.workload.fnirs_subset_select import _effective_n, _fit, _knee


def test_effective_n_bounds():
    k = 5
    assert abs(_effective_n(np.full(k, 1 / k)) - k) < 1e-9        # uniform -> K effective families
    assert _effective_n(np.array([1.0, 0, 0, 0, 0])) < 1.001      # all mass on one -> ~1
    assert _effective_n(np.array([0.9, 0.1])) < _effective_n(np.array([0.5, 0.5]))  # concentrated < spread


def test_knee_picks_utopia_corner():
    pts = [{"acc": 0.40, "eff_n": 1.5}, {"acc": 0.47, "eff_n": 3.0}, {"acc": 0.475, "eff_n": 8.0}]
    assert _knee(pts)["eff_n"] == 3.0                             # best accuracy-per-feature tradeoff


def test_weighted_head_learns_to_prefer_informative_group():
    import torch
    rng = np.random.default_rng(0)
    n, ch = 240, 4
    # 2 groups of `ch` columns: group 0 carries the class (mean offset), group 1 is pure noise
    y = rng.integers(0, 3, n)
    g0 = (y[:, None] + rng.standard_normal((n, ch)) * 0.3).astype(np.float32)
    g1 = rng.standard_normal((n, ch)).astype(np.float32)
    X = np.concatenate([g0, g1], axis=1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gi = torch.as_tensor([0] * ch + [1] * ch, dtype=torch.long, device=dev)
    m = _fit(X, y, gi, 2, 3, lam=0.0, hp={"lr": 0.05, "weight_decay": 1e-4, "epochs": 300})
    w = m.weights().detach().cpu().numpy()
    assert w[0] > w[1]                                            # informative group gets more weight
