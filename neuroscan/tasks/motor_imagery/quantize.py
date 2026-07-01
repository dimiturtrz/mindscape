"""Stage-2 tail: train our decoder, then quantize it for the edge and report the honest deployment triad.

    python -m neuroscan.tasks.motor_imagery.quantize --method atcnet --resample 250 --fmin 4 --fmax 40

Trains one within-subject model (the realistic edge artifact — a decoder calibrated to its user), exports
the crop-sized net to ONNX, dynamically quantizes to INT8, and reports fp32-vs-int8 accuracy / size / CPU
latency. The contribution isn't a smaller model — it's the *measured cost* of making it deployable (and,
for these tiny EEG nets, the honest finding that they're already edge-sized).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from core import export_onnx
from core.data import store
from core.data.eeg.base import EpochCfg
from neuroscan import tracking
from neuroscan.evaluation import metrics
from neuroscan.models import decoders


def _onnx_trial_proba(path, X_std, crop_len, n_test_crops):
    """Run an ONNX model over the crops of each trial and average softmax back per trial."""
    if crop_len:
        Xc, tidx = decoders._crops(X_std, crop_len, n_test_crops)
    else:
        Xc, tidx = X_std, np.arange(len(X_std))
    logits = export_onnx.run(path, Xc)
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z); p /= p.sum(1, keepdims=True)
    out = np.zeros((len(X_std), p.shape[1]))
    np.add.at(out, tidx, p)
    return out / (n_test_crops if crop_len else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="bnci2014_001")
    ap.add_argument("--method", default="atcnet", choices=sorted(decoders.MODELS))
    ap.add_argument("--subject", default=None, help="subject to train the edge model on (default: first)")
    ap.add_argument("--test-session", default="1test")
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--fmin", type=float, default=4.0)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--out", default="runs/quantize")
    args = ap.parse_args()

    meta = store.load(args.dataset, EpochCfg(resample=args.resample, fmin=args.fmin, fmax=args.fmax))
    sub = args.subject or sorted(meta["subject"].unique().to_list())[0]
    one = meta.filter(pl.col("subject") == str(sub))
    Xtr, ytr = store.gather(one.filter(pl.col("session") != args.test_session))
    Xte, yte = store.gather(one.filter(pl.col("session") == args.test_session))
    print(f"subject {sub}: train {Xtr.shape} -> test {Xte.shape}")

    fit, _ = decoders.make(args.method)
    clf = fit(Xtr, ytr)
    crop_len = clf.crop_len
    n_chans = Xtr.shape[1]
    n_times = crop_len or Xtr.shape[2]

    acc_fp32_torch = metrics.accuracy(yte, clf.predict_proba(Xte).argmax(1))   # crop-aware
    Xte_std = clf.std(Xte)

    out = Path(args.out)
    fp32_path = out / f"{args.method}_sub{sub}_fp32.onnx"
    int8_path = out / f"{args.method}_sub{sub}_int8.onnx"
    export_onnx.export(clf.net, n_chans, n_times, fp32_path, device="cpu")

    # parity gate on a batch of crops (the unit the net actually consumes)
    Xc = decoders._crops(Xte_std, crop_len, clf.n_test_crops)[0] if crop_len else Xte_std
    gap = export_onnx.parity(clf.net, fp32_path, Xc[:128], device="cpu")
    assert gap < 1e-3, f"ONNX parity failed: max|Δlogit| = {gap:.2e}"

    acc_fp32 = metrics.accuracy(yte, _onnx_trial_proba(fp32_path, Xte_std, crop_len, clf.n_test_crops).argmax(1))

    rep = {"method": args.method, "subject": str(sub), "n_chans": n_chans, "crop_len": crop_len,
           "parity_max_dlogit": gap,
           "accuracy": {"torch_fp32": acc_fp32_torch, "onnx_fp32": acc_fp32},
           "size_mb": {"fp32": export_onnx.file_mb(fp32_path)},
           "latency_ms_cpu": {"fp32": export_onnx.latency_ms(fp32_path, Xc)}}
    try:
        export_onnx.quantize_int8(fp32_path, int8_path)
        rep["accuracy"]["onnx_int8"] = metrics.accuracy(
            yte, _onnx_trial_proba(int8_path, Xte_std, crop_len, clf.n_test_crops).argmax(1))
        rep["size_mb"]["int8"] = export_onnx.file_mb(int8_path)
        rep["size_mb"]["ratio"] = round(rep["size_mb"]["fp32"] / max(rep["size_mb"]["int8"], 1e-6), 2)
        rep["latency_ms_cpu"]["int8"] = export_onnx.latency_ms(int8_path, Xc)
        rep["latency_ms_cpu"]["speedup"] = round(
            rep["latency_ms_cpu"]["fp32"] / max(rep["latency_ms_cpu"]["int8"], 1e-6), 2)
    except Exception as e:
        rep["int8_error"] = str(e)

    rep_dir = out / f"{args.method}_sub{sub}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (out / f"{args.method}_sub{sub}.json").write_text(json.dumps(rep, indent=2))
    with tracking.run("mindscape", f"quantize_{args.method}",
                      params={"method": args.method, "subject": str(sub)},
                      tags={"kind": "quantize"}, run_dir=rep_dir):
        a, s, l = rep["accuracy"], rep["size_mb"], rep["latency_ms_cpu"]
        m = {"acc_torch_fp32": a["torch_fp32"], "acc_onnx_fp32": a["onnx_fp32"],
             "size_fp32_mb": s["fp32"], "latency_fp32_ms": l["fp32"], "parity_max_dlogit": gap}
        if "onnx_int8" in a:
            m.update({"acc_onnx_int8": a["onnx_int8"], "size_int8_mb": s["int8"], "latency_int8_ms": l["int8"]})
        tracking.metrics(m)
    print(f"\n=== {args.method} edge quantization (subject {sub}) ===")
    print(f"  parity max|Δlogit| {gap:.2e}  (gate < 1e-3) OK")
    a = rep["accuracy"]
    print(f"  accuracy  torch {a['torch_fp32']:.3f} | onnx-fp32 {a['onnx_fp32']:.3f}"
          + (f" | onnx-int8 {a['onnx_int8']:.3f}" if "onnx_int8" in a else " | int8 N/A"))
    print(f"  size MB   fp32 {rep['size_mb']['fp32']}"
          + (f" -> int8 {rep['size_mb']['int8']} ({rep['size_mb']['ratio']}x)" if "int8" in rep["size_mb"] else ""))
    lat = rep["latency_ms_cpu"]
    print(f"  latency   fp32 {lat['fp32']} ms" + (f" -> int8 {lat['int8']} ms ({lat['speedup']}x)" if "int8" in lat else ""))
    print(f"-> {out}/{args.method}_sub{sub}.json")


if __name__ == "__main__":
    main()
