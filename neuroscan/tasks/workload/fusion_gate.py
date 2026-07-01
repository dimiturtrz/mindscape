"""Stage-3: a compact INPUT-level gated fusion — the one path left after output-space fusion was closed.

The complementarity is real (oracle 0.688 vs best-single 0.474, near-independent errors) but no output-space
combiner cashes it, because a decoder's confidence does not predict its correctness (run_fusion.py). So the
per-trial "which modality to trust" signal has to be learned from the INPUT features, not the decisions.

This is the smallest model that can do that: shallow per-modality encoders → a gate network that reads BOTH
embeddings and emits a per-trial mixing weight α, → α·p_eeg + (1−α)·p_fnirs. Everything is sized for n=26 /
~700 blocks (the SOTA review's hard constraint): d_model 16–32, dropout ≥ 0.5, weight decay, per-subject
z-scoring, and — decisively — a nested split (inner val over TRAIN subjects for early stopping) under the
same outer 5-fold GroupKFold as every other fusion number.

RESULT — an honest NEGATIVE (kept as an artifact). The gate scores ~0.573 fold-mean, which looks like a
+0.10 win over fNIRS-alone (0.474) but is NOT one: an ablation shows per-subject z-scoring alone lifts
EEG-band-power → LDA to 0.581 (from 0.407 raw; the absolute band-power was subject-idiosyncratic), and the
gate merely *ties* that best single modality (z-EEG 0.581, z-concat-LDA 0.578, late-z 0.575, gate 0.573). So
the learned output-mixing gate captures none of the oracle headroom either — consistent with run_fusion's
finding that the per-trial reliability signal is not in the probabilities. The real transfer lever the
investigation surfaced is the per-subject normalization (see mindscape-lpv follow-up), not the fusion.

    python -m neuroscan.tasks.workload.fusion_gate --no-record   # 5-fold GroupKFold (negative result; not recorded)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from baselines.eeg_bandpower import _bandpower
from baselines.fnirs_features import _features
from core.data import store
from core.data.eeg.base import EpochCfg
from core.data.fnirs.base import FnirsCfg
from neuroscan.evaluation import metrics

_EEG, _FNIRS = "shin2017_nback_eeg", "shin2017_nback"
_EEG_CFG = EpochCfg(fmin=4, fmax=30, tmin=0.0, tmax=40.0, resample=100.0)
_FNIRS_CFG = FnirsCfg()
_SEED = 0


def _load_features():
    """Return per-block EEG band-power + fNIRS mean/slope/peak features, the label, and the subject id —
    block-aligned across the two modalities (hard guard on the label sequence)."""
    me = store.load(_EEG, _EEG_CFG)
    mf = store.load(_FNIRS, _FNIRS_CFG)
    subs = sorted(set(me["subject"].unique().to_list()) & set(mf["subject"].unique().to_list()))
    qe = me.filter(me["subject"].is_in(subs))
    qf = mf.filter(mf["subject"].is_in(subs))
    Xe, ye = store.gather(qe)
    Xf, yf = store.gather(qf)
    assert np.array_equal(ye, yf), "EEG/fNIRS blocks misaligned — fusion invalid"
    g = qe["subject"].to_numpy()
    Fe = _bandpower(Xe, _EEG_CFG.resample).astype(np.float32)          # [n, 28*3]
    Ff = _features(Xf).astype(np.float32)                              # [n, ch*3]
    return Fe, Ff, ye.astype(np.int64), g


def _zscore_per_subject(F, g):
    """Standardize each feature within each subject (its own mean/std) — unsupervised, so it applies to a
    held-out test subject too. Removes the subject-specific offset that sinks cross-subject band-power."""
    out = np.empty_like(F)
    for s in np.unique(g):
        m = g == s
        mu, sd = F[m].mean(0), F[m].std(0)
        out[m] = (F[m] - mu) / (sd + 1e-6)
    return out


class _GatedFusion:
    """Shallow per-modality encoders + a gate that mixes their class-probabilities per trial. Kept tiny and
    heavily regularized; trained with early stopping on an inner validation split."""

    def __init__(self, d_e, d_f, n_classes=3, d_model=16, dropout=0.5, wd=1e-2, lr=3e-3, max_epochs=200,
                 patience=20):
        self.cfg = dict(d_e=d_e, d_f=d_f, n_classes=n_classes, d_model=d_model, dropout=dropout,
                        wd=wd, lr=lr, max_epochs=max_epochs, patience=patience)

    def _build(self):
        import torch.nn as nn

        c = self.cfg
        d = c["d_model"]

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                enc = lambda din: nn.Sequential(nn.Linear(din, d), nn.ReLU(), nn.Dropout(c["dropout"]))
                s.enc_e, s.enc_f = enc(c["d_e"]), enc(c["d_f"])
                s.head_e = nn.Linear(d, c["n_classes"])
                s.head_f = nn.Linear(d, c["n_classes"])
                s.gate = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Dropout(c["dropout"]),
                                       nn.Linear(d, 1))          # per-trial scalar α (pre-sigmoid)

            def forward(s, xe, xf):
                import torch
                ze, zf = s.enc_e(xe), s.enc_f(xf)
                pe = torch.softmax(s.head_e(ze), dim=1)
                pf = torch.softmax(s.head_f(zf), dim=1)
                a = torch.sigmoid(s.gate(torch.cat([ze, zf], dim=1)))  # [n,1] in (0,1)
                p = a * pe + (1 - a) * pf
                return p, a.squeeze(1)

        return Net()

    def fit(self, Xe, Xf, y, Xe_va, Xf_va, y_va):
        import torch

        torch.manual_seed(_SEED)
        self.net = self._build()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg["lr"], weight_decay=self.cfg["wd"])
        nll = torch.nn.NLLLoss()
        te, tf, ty = map(torch.as_tensor, (Xe, Xf, y))
        ve, vf, vy = map(torch.as_tensor, (Xe_va, Xf_va, y_va))
        best, best_state, bad = 1e9, None, 0
        for _ep in range(self.cfg["max_epochs"]):
            self.net.train(); opt.zero_grad()
            p, _ = self.net(te, tf)
            loss = nll(torch.log(p + 1e-12), ty)
            loss.backward(); opt.step()
            self.net.eval()
            with torch.no_grad():
                pv, _ = self.net(ve, vf)
                vloss = nll(torch.log(pv + 1e-12), vy).item()
            if vloss < best - 1e-4:
                best, best_state, bad = vloss, {k: v.clone() for k, v in self.net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.cfg["patience"]:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, Xe, Xf):
        import torch

        self.net.eval()
        with torch.no_grad():
            p, a = self.net(torch.as_tensor(Xe), torch.as_tensor(Xf))
        return p.numpy(), a.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold

    Fe, Ff, y, g = _load_features()
    Fe, Ff = _zscore_per_subject(Fe, g), _zscore_per_subject(Ff, g)
    subs = np.array(sorted(set(g)))
    n_classes = int(y.max()) + 1
    print(f"gated fusion: {len(y)} blocks · {len(subs)} subjects · EEG {Fe.shape[1]}d · fNIRS {Ff.shape[1]}d · "
          f"chance {1/n_classes:.3f}")

    rows, P, A, Y = [], [], [], []
    outer = GroupKFold(n_splits=args.k)
    for i, (tr, te) in enumerate(outer.split(subs, groups=subs)):
        tr_subs, te_subs = subs[tr], subs[te]
        # inner val: hold out one GroupKFold slice of the TRAIN subjects for early stopping
        itr, iva = next(GroupKFold(n_splits=4).split(tr_subs, groups=tr_subs))
        va_subs = tr_subs[iva]; fit_subs = tr_subs[itr]
        m_fit = np.isin(g, fit_subs); m_va = np.isin(g, va_subs); m_te = np.isin(g, te_subs)

        clf = _GatedFusion(Fe.shape[1], Ff.shape[1], n_classes)
        clf.fit(Fe[m_fit], Ff[m_fit], y[m_fit], Fe[m_va], Ff[m_va], y[m_va])
        p, a = clf.predict(Fe[m_te], Ff[m_te])
        P.append(p); A.append(a); Y.append(y[m_te])
        acc = metrics.accuracy(y[m_te], p.argmax(1))
        rows.append({"fold": str(i), "n": int(m_te.sum()), "gate_acc": acc,
                     "alpha_mean": float(a.mean())})
        print(f"  fold{i}: gate {acc:.3f} | ᾱ(eeg-weight) {a.mean():.2f} (n={int(m_te.sum())})")

    y_all, P_all = np.concatenate(Y), np.concatenate(P)
    gate = float((P_all.argmax(1) == y_all).mean())
    fold_mean = float(np.mean([r["gate_acc"] for r in rows]))
    std = float(np.std([r["gate_acc"] for r in rows]))
    print(f"\n=== gated fusion · 5-fold GroupKFold · shin n-back ===")
    print(f"  gate pooled {gate:.3f} | fold-mean {fold_mean:.3f} ± {std:.3f}")
    print(f"  NOTE: this ~{fold_mean:.2f} is NOT a fusion win — it ties z-scored-EEG-alone (~0.581) and "
          "z-concat-LDA (~0.578).")
    print("  The lift over raw fNIRS (0.474) is per-subject z-scoring rescuing EEG (0.407->0.581), NOT the "
          "gate; the gate captures no oracle headroom (see run_fusion + the ablation).")

    run_dir = Path(args.out) if args.out else Path("runs") / "fusion_gate_cross_subject_kfold_shin2017_nback"
    run_dir.mkdir(parents=True, exist_ok=True)
    res = {"method": "fusion_gate", "regime": "cross_subject_kfold", "n_classes": n_classes,
           "fold_mean": {"acc": fold_mean}, "per_role_mean": {"gate": fold_mean},
           "pooled_acc": gate, "acc_std": std, "per_fold": rows}
    (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
    if not args.no_record:
        from neuroscan.evaluation import results
        results.record(run_dir)
    print(f"-> {run_dir}/aggregate.json")


if __name__ == "__main__":
    main()
