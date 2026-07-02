"""Riemannian transfer methods (the RPA ladder): per-domain alignment moves each domain's covariance mean
to the identity. Synthetic covariances — no dataset, no experiment scaffolding."""
import numpy as np

from baselines.eeg import transfer


def _domain(n, d, t, shift, rng):
    """n covariance-bearing trials whose cloud is displaced by an SPD `shift`."""
    return np.stack([shift @ (B @ B.T / t) @ shift for B in rng.normal(size=(n, d, t))])


def test_align_domains_centers_each_source_domain_and_target_to_identity():
    from pyriemann.utils.mean import mean_riemann
    rng = np.random.default_rng(0)
    d = 4
    sa = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))   # source domain A's location
    sb = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))   # source domain B's, elsewhere
    st = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))   # target's, elsewhere again
    Csrc = np.concatenate([_domain(25, d, 12, sa, rng), _domain(25, d, 12, sb, rng)])
    groups = np.array(["a"] * 25 + ["b"] * 25)
    Cte = _domain(25, d, 12, st, rng)

    Cs, Ct = transfer.align_domains(Csrc, groups, Cte, scale=False)
    # every domain (both source subjects AND the unsupervised target) lands on I after its own re-centering
    assert np.allclose(mean_riemann(Cs[groups == "a"]), np.eye(d), atol=1e-4)
    assert np.allclose(mean_riemann(Cs[groups == "b"]), np.eye(d), atol=1e-4)
    assert np.allclose(mean_riemann(Ct), np.eye(d), atol=1e-4)


def test_align_domains_recenters_each_TARGET_subject_separately():
    """With multiple target subjects (a k-fold test fold), `target_groups` must re-center each on its OWN
    mean — pooling them (target_groups=None) would align to a shared blob and NOT put each subject on I."""
    from pyriemann.utils.mean import mean_riemann
    rng = np.random.default_rng(3)
    d = 4
    st1 = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))
    st2 = (lambda A: A @ A.T + 4 * np.eye(d))(rng.normal(size=(d, d)))
    Csrc = _domain(20, d, 12, np.eye(d), rng)
    Cte = np.concatenate([_domain(20, d, 12, st1, rng), _domain(20, d, 12, st2, rng)])
    tg = np.array(["t1"] * 20 + ["t2"] * 20)

    _, Ct = transfer.align_domains(Csrc, np.array(["s"] * 20), Cte, scale=False, target_groups=tg)
    assert np.allclose(mean_riemann(Ct[tg == "t1"]), np.eye(d), atol=1e-4)   # each target subject on I
    assert np.allclose(mean_riemann(Ct[tg == "t2"]), np.eye(d), atol=1e-4)
    # pooled (target_groups=None) does NOT land both subjects on I
    _, Ct_pool = transfer.align_domains(Csrc, np.array(["s"] * 20), Cte, scale=False)
    assert not np.allclose(mean_riemann(Ct_pool[tg == "t1"]), np.eye(d), atol=1e-2)
