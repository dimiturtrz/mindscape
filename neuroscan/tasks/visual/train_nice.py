"""Train + evaluate the NICE EEG->image baseline on THINGS-EEG2.

Pipeline: adapter epochs (our own preprocessing off the raw) -> NICE encoder -> InfoNCE against the viewed
image's CLIP embedding -> zero-shot retrieval on the 200 held-out test concepts. The honest number is the
CROSS-SUBJECT single-trial top-k (train subjects != test subject); within-subject and repeat-averaged are
reported alongside as the (inflated) references the field usually quotes.

    # within-subject smoke (one subject):
    python -m neuroscan.tasks.visual.train_nice --train 1 --test 1 --epochs 20
    # cross-subject (the headline; needs >=2 subjects downloaded):
    python -m neuroscan.tasks.visual.train_nice --train 1 2 3 --test 4 --epochs 40

Test concepts are disjoint from training images by dataset design, so retrieval is zero-shot in both regimes;
"cross-subject" additionally holds out the *person*. Chance = 1/200 = 0.5%.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.data.eeg import things_eeg2 as te
from neuroscan.models.nice import NiceEncoder, clip_infonce, retrieval_topk
from neuroscan.tasks.visual import clip_targets as ct


def _targets(img_files: np.ndarray, split: str) -> np.ndarray:
    """CLIP embedding per epoch, looked up by the image the subject viewed."""
    by_file = ct.embeddings_by_file(split)
    return np.stack([by_file[f] for f in img_files]).astype(np.float32)


def _load(subjects: list[int], split: str, resample: float):
    X, concept, files, _ = te.get_epochs(subjects, split=split, resample=resample)
    return X, concept, _targets(files, split)


@torch.no_grad()
def evaluate(enc, Xte, concept_te, protos, device, batch: int = 512) -> dict:
    """Single-trial + concept-averaged retrieval top-1/5 against the 200-concept prototype bank.

    Eval batch is capped at 512: batches >=2048 trip a cuDNN illegal-memory-access on this conv shape
    (Blackwell / cu130) — reproduced with random input, independent of the data.
    """
    enc.eval()
    z = []
    for i in range(0, len(Xte), batch):
        z.append(enc(torch.tensor(Xte[i:i + batch]).to(device)).cpu())
    z = torch.cat(z)                                                    # [N,512] normalized
    cand = torch.tensor(protos)
    lab = torch.tensor(concept_te)
    single = retrieval_topk(z, cand, lab)
    # concept-averaged: mean EEG embedding per concept, re-normalized
    n = int(concept_te.max()) + 1
    zc = torch.stack([torch.nn.functional.normalize(z[lab == c].mean(0), dim=-1) for c in range(n)])
    avg = retrieval_topk(zc, cand, torch.arange(n))
    return {"single_trial": single, "concept_avg": avg}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=int, nargs="+", default=[1], help="training subject id(s)")
    ap.add_argument("--test", type=int, default=1, help="held-out test subject id")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512,
                    help="<=1024; batch>=2048 trips a cuDNN illegal-access on this conv shape (Blackwell/cu130)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=8, help="early-stop patience on val-top1 (epochs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    regime = "within" if args.train == [args.test] else "cross"
    print(f"regime={regime}  train={args.train}  test={args.test}  device={device}")

    Xtr, ctr, Ytr = _load(args.train, "training", args.resample)
    Xte, cte, _ = _load([args.test], "test", args.resample)
    protos = ct.concept_prototypes("test")
    print(f"train {Xtr.shape} -> CLIP {Ytr.shape} | test {Xte.shape} ({int(cte.max())+1} concepts)")

    # Leak-free early stopping: hold out 10% of TRAINING concepts as a validation retrieval set. Model
    # selection (which epoch to keep) reads ONLY this val set — the test subject/concepts are never used to
    # pick the checkpoint, so the reported test number isn't inflated by selecting the best test epoch.
    rng = np.random.default_rng(args.seed)
    concepts = np.unique(ctr)
    val_set = set(rng.choice(concepts, max(1, len(concepts) // 10), replace=False).tolist())
    vmask = np.array([c in val_set for c in ctr])
    val_list = sorted(val_set)
    c2i = {c: i for i, c in enumerate(val_list)}
    vproto = np.stack([Ytr[ctr == c].mean(0) for c in val_list]).astype(np.float32)
    vproto /= (np.linalg.norm(vproto, axis=1, keepdims=True) + 1e-8)
    Xval, cval = Xtr[vmask], np.array([c2i[c] for c in ctr[vmask]])
    Xtr_f, Ytr_f = Xtr[~vmask], Ytr[~vmask]
    print(f"early-stop val: {len(val_list)} held-out train concepts, {int(vmask.sum())} epochs")

    enc = NiceEncoder(Xtr.shape[1], Xtr.shape[2], embed_dim=Ytr.shape[1]).to(device)
    logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([*enc.parameters(), logit_scale], lr=args.lr, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr_f), torch.tensor(Ytr_f)),
                    batch_size=args.batch, shuffle=True, drop_last=True)

    best_val, best_state, best_ep, since = -1.0, None, -1, 0
    for ep in range(args.epochs):
        enc.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), torch.nn.functional.normalize(yb.to(device), dim=-1)
            opt.zero_grad()
            loss = clip_infonce(enc(xb), yb, logit_scale.exp().clamp(max=100))
            loss.backward()
            opt.step()
            tot += loss.item()
        val = evaluate(enc, Xval, cval, vproto, device)["single_trial"][1]     # selection signal (val only)
        if val > best_val:
            best_val, best_ep, since = val, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}
        else:
            since += 1
        if ep % 5 == 0 or ep == args.epochs - 1:
            r = evaluate(enc, Xte, cte, protos, device)
            print(f"ep {ep:3d}  loss {tot/len(dl):.3f}  val-top1 {val*100:.2f}%  "
                  f"test single {r['single_trial'][1]*100:.2f}%/{r['single_trial'][5]*100:.2f}%  "
                  f"avg {r['concept_avg'][1]*100:.2f}%/{r['concept_avg'][5]*100:.2f}%")
        if since >= args.patience:
            print(f"early stop at ep {ep} (best val = ep {best_ep})")
            break

    enc.load_state_dict(best_state)                            # report TEST at the best-VAL checkpoint
    r = evaluate(enc, Xte, cte, protos, device)
    result = {"regime": regime, "train": args.train, "test": args.test,
              "best_val_epoch": best_ep, "val_top1": best_val, "epochs_run": ep + 1,
              "chance_top1": 1 / (int(cte.max()) + 1), **r}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
