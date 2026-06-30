"""Stage-0 entrypoint: decode a motor-imagery dataset under a chosen regime, through the eval harness.

    python -m neuroscan.experiments.run --method csp_lda --regime within
    python -m neuroscan.experiments.run --method atcnet  --regime within --resample 250
    python -m neuroscan.experiments.run --method eegnet  --regime cross_subject

Methods: csp_lda (the baseline ceiling) + the braindecode decoders (eegnet, shallow_fbcsp, deep4,
atcnet, eegconformer) — commodity → near-SOTA. The within/cross_subject contrast IS the headline:
the gap between them is the honest out-of-distribution number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core import reference
from core.data import store
from core.data.eeg.base import EpochCfg
from neuroscan import models
from neuroscan.evaluation import harness


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--method", default="csp_lda", choices=models.method_names())
    ap.add_argument("--regime", default="within", choices=["within", "cross_subject"])
    ap.add_argument("--test-session", default=None,
                    help="within-subject: hold out this session as test (the standard 2a protocol)")
    ap.add_argument("--resample", type=float, default=128.0,
                    help="epoch resample rate (Hz); strong nets prefer native 250")
    ap.add_argument("--fmin", type=float, default=8.0, help="band low cut (Hz); 4 = broadband for DL")
    ap.add_argument("--fmax", type=float, default=32.0, help="band high cut (Hz); 40 = broadband for DL")
    ap.add_argument("--out", default=None, help="write aggregate.json here")
    args = ap.parse_args()

    cfg = EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax)
    meta = store.load(args.dataset, cfg)
    print(f"cloud: {len(meta)} epochs · {meta['subject'].n_unique()} subjects · "
          f"sessions {sorted(meta['session'].unique().to_list())} · recipe {cfg.key()}")

    test_sessions = [args.test_session] if (args.regime == "within" and args.test_session) else ()
    folds = harness.folds_for(meta, args.regime, test_sessions=test_sessions)
    fit_fn, score_fn = models.get_method(args.method)
    n_classes = 4

    run_dir = Path(args.out) if args.out else Path("runs") / f"{args.method}_{args.regime}_{args.dataset}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {args.method} · {args.regime} · {args.dataset} ({len(folds)} folds) ===")
    res = harness.run(args.method, fit_fn, score_fn, folds, n_classes, regime=args.regime,
                      params={"method": args.method, "regime": args.regime,
                              "dataset": args.dataset, "resample": args.resample},
                      run_dir=run_dir)

    out = run_dir / "aggregate.json"
    out.write_text(json.dumps(res, indent=2))
    from neuroscan.evaluation import modelcard
    modelcard.write(res, args.dataset, args.regime, run_dir / "CARD.md")
    ref_regime = "within_subject" if args.regime == "within" else "cross_subject"
    print(f"\nfold-mean acc {res['fold_mean']['acc']:.3f} | pooled acc {res['pooled']['acc']:.3f} "
          f"| ece {res['fold_mean']['ece']:.3f}")
    print("  vs reference: " + reference.compare(res["fold_mean"]["acc"], args.dataset, ref_regime, args.method))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
