"""Stage-2 entrypoint: decode fNIRS n-back workload through the SAME eval harness as EEG.

Different task/modality (Shin n-back, 3-class, hemodynamic), same spine — proving the harness is
modality-agnostic. The right decoder differs though: covariance methods (csp_lda, riemann) sit at chance
because fNIRS class info is in the HbO amplitude the covariance discards; `fnirs_lda` (mean+slope+peak ->
LDA) is the field-standard that actually reads it.

    python -m neuroscan.experiments.run_fnirs --method fnirs_lda --regime cross_subject
    python -m neuroscan.experiments.run_fnirs --method fnirs_lda --regime within --test-session 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.data import store
from core.data.fnirs.base import FnirsCfg
from neuroscan import models
from neuroscan.evaluation import harness


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="shin2017_nback")
    ap.add_argument("--method", default="fnirs_lda", choices=models.method_names())
    ap.add_argument("--regime", default="cross_subject", choices=["within", "cross_subject", "cross_subject_kfold"])
    ap.add_argument("--test-session", default="2", help="within-subject: session held out as test")
    ap.add_argument("--l-freq", type=float, default=0.01)
    ap.add_argument("--h-freq", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-record", action="store_true",
                    help="skip updating the committed results.json snapshot (scratch/experimental runs)")
    args = ap.parse_args()

    cfg = FnirsCfg(l_freq=args.l_freq, h_freq=args.h_freq)
    meta = store.load(args.dataset, cfg)
    n_classes = int(meta["label_id"].max()) + 1
    chance = 1.0 / n_classes
    print(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · "
          f"{n_classes} classes {sorted(meta['label'].unique().to_list())} · recipe {cfg.key()}")

    test_sessions = [args.test_session] if (args.regime == "within" and args.test_session) else ()
    folds = harness.folds_for(meta, args.regime, test_sessions=test_sessions)
    fit_fn, score_fn = models.get_method(args.method)

    run_dir = Path(args.out) if args.out else Path("runs") / f"{args.method}_{args.regime}_{args.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {args.method} · {args.regime} · {args.dataset} ({len(folds)} folds, chance {chance:.3f}) ===")
    n_jobs = -1 if args.method in {"csp_lda", "riemann", "riemann_acm", "fnirs_lda"} else 1
    res = harness.run(args.method, fit_fn, score_fn, folds, n_classes, regime=args.regime,
                      params={"method": args.method, "regime": args.regime, "dataset": args.dataset,
                              "modality": "fnirs"}, run_dir=run_dir, n_jobs=n_jobs)
    (run_dir / "aggregate.json").write_text(json.dumps(res, indent=2))
    from neuroscan.evaluation import results
    if not args.no_record and results.record(run_dir):
        print(f"   recorded -> results.json ({run_dir.name})")
    fm = res["fold_mean"]
    print(f"\nfold-mean acc {fm['acc']:.3f} | kappa {fm['kappa']:.3f} | ece {fm['ece']:.3f}  "
          f"(chance {chance:.3f})")
    print(f"-> {run_dir}/aggregate.json")


if __name__ == "__main__":
    main()
