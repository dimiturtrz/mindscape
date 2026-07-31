"""Riemannian transfer methods (the RPA ladder): per-domain alignment moves each domain's covariance mean
to the identity. Synthetic covariances — no dataset, no experiment scaffolding."""
import numpy as np

from baselines.eeg import transfer
from baselines.eeg.transfer import Domain


def _domain(n, d, t, shift, rng):
    """n covariance-bearing trials whose cloud is displaced by an SPD `shift`."""
    return np.stack([shift @ (B @ B.T / t) @ shift for B in rng.normal(size=(n, d, t))])


def _class_cov(rng, d, n, cls, t=20):
    """SPD covariances with a class signal in channel-0 variance (class 1 = higher) — a separable contrast
    that survives per-subject re-centering. `C = B B^T / t + 0.5 I` (PSD + PD -> SPD)."""
    B = rng.normal(size=(n, d, t))
    if cls == 1:
        B[:, 0, :] *= 2.5
    return np.stack([b @ b.T / t + 0.5 * np.eye(d) for b in B])


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


def test_recentered_tangent_features_shape_and_finite():
    """Per-subject re-center -> tangent at I -> a Euclidean vector `[n, d(d+1)/2]`, finite."""
    rng = np.random.default_rng(0)
    d = 4
    C = np.concatenate([_class_cov(rng, d, 15, 0), _class_cov(rng, d, 15, 1)])
    groups = np.array(["a"] * 15 + ["b"] * 15)             # two subjects, re-centered independently
    F = transfer.recentered_tangent_features(C, groups)
    assert F.shape == (30, d * (d + 1) // 2)               # symmetric tangent vector length
    assert np.isfinite(F).all()


def _two_subject_source(rng, d):
    """Two labelled source subjects, each with both classes; returns (cov, labels, groups)."""
    cov, lab, grp = [], [], []
    for s in ("s0", "s1"):
        for cls in (0, 1):
            cov.append(_class_cov(rng, d, 15, cls))
            lab.extend([cls] * 15)
            grp.extend([s] * 15)
    return np.concatenate(cov), np.array(lab), np.array(grp)


def test_zero_shot_predict_probs_decode_shared_contrast():
    rng = np.random.default_rng(1)
    d = 4
    Csrc, ysrc, gsrc = _two_subject_source(rng, d)
    yte = np.array([0] * 15 + [1] * 15)
    Cte = np.concatenate([_class_cov(rng, d, 15, 0), _class_cov(rng, d, 15, 1)])
    source = Domain(cov=Csrc, labels=ysrc, groups=gsrc)
    target = Domain(cov=Cte, groups=np.array(["t"] * 30))
    for scale in (False, True):
        probs = transfer.zero_shot_predict(source, target, scale=scale)
        assert probs.shape == (30, 2)
        assert np.allclose(probs.sum(1), 1.0, atol=1e-6)
        assert (probs.argmax(1) == yte).mean() > 0.7       # shared class contrast transfers zero-shot


def test_calibrated_predict_returns_valid_int_labels_both_kinds():
    rng = np.random.default_rng(2)
    d = 4
    Csrc, ysrc, gsrc = _two_subject_source(rng, d)
    calib_cov = np.concatenate([_class_cov(rng, d, 10, 0), _class_cov(rng, d, 10, 1)])
    calib_lab = np.array([0] * 10 + [1] * 10)
    eval_cov = np.concatenate([_class_cov(rng, d, 8, 0), _class_cov(rng, d, 8, 1)])
    source = Domain(cov=Csrc, labels=ysrc, groups=gsrc)
    calib = Domain(cov=calib_cov, labels=calib_lab)
    evaluation = Domain(cov=eval_cov)
    for kind in ("rpa", "mdwm"):
        pred = transfer.calibrated_predict(kind, source, calib, evaluation)
        assert pred.shape == (16,)
        assert pred.dtype.kind == "i"                      # int labels
        assert set(pred.tolist()) <= {0, 1}                # valid class ids
