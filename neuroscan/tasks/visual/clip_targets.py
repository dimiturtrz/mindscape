"""CLIP image embeddings — the EEG->image retrieval targets for the NICE baseline.

The EEG encoder is trained to match the CLIP embedding of the viewed image (contrastive), and evaluated by
retrieving among the 200 test concepts' embeddings. So the image side is fixed, precomputed once, and cached:
this module walks the THINGS-EEG2 image set (`<data>/raw/things_eeg2/images/{training,test}_images/`) and
writes per-image CLIP embeddings + their concept index to `<data>/processed/things_eeg2/clip_<split>.npz`.

Concept index = position in the sorted concept-folder list (00001_aardvark -> 0, ...), stable across the EEG
adapter and here, so an epoch's concept id lines up with these embeddings. Model: open_clip ViT-B/32 (the
NICE target); embeddings are L2-normalized (cosine == dot).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import processed_dir, raw_dir

_ROOT = "things_eeg2"
_CLIP = ("ViT-B-32", "laion2b_s34b_b79k")   # open_clip arch + pretrained tag (NICE uses a CLIP ViT)
_EXTS = (".jpg", ".jpeg", ".png")


def _img_root(split: str) -> Path:
    return raw_dir() / _ROOT / "images" / f"{split}_images"


def concept_dirs(split: str) -> list[Path]:
    """Sorted concept folders for a split (the sort defines the concept index used everywhere)."""
    return sorted(p for p in _img_root(split).iterdir() if p.is_dir())


def _list_images(split: str) -> tuple[list[Path], list[int]]:
    """(image paths, concept index per image) over a split, in concept-sorted then filename-sorted order."""
    paths: list[Path] = []
    concepts: list[int] = []
    for ci, d in enumerate(concept_dirs(split)):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in _EXTS:
                paths.append(f)
                concepts.append(ci)
    return paths, concepts


def _load_clip(device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(_CLIP[0], pretrained=_CLIP[1])
    model = model.eval().to(device)
    return model, preprocess


def compute(split: str, *, device: str | None = None, batch: int = 256, force: bool = False) -> Path:
    """Embed every image in `split` with CLIP, cache to `<processed>/things_eeg2/clip_<split>.npz`.

    npz: `emb` [N, 512] float32 (L2-normalized), `concept` [N] int (concept index), `paths` [N] str.
    Idempotent — returns the cache path, recomputing only if missing or `force`.
    """
    import torch
    from PIL import Image

    out = processed_dir() / _ROOT / f"clip_{split}.npz"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = _load_clip(device)

    paths, concepts = _list_images(split)
    embs = np.empty((len(paths), 512), dtype=np.float32)
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        px = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
        with torch.no_grad():
            z = model.encode_image(px).float()
            z = z / z.norm(dim=-1, keepdim=True)
        embs[i:i + len(chunk)] = z.cpu().numpy()
        print(f"[clip:{split}] {i + len(chunk)}/{len(paths)}")
    np.savez(out, emb=embs, concept=np.asarray(concepts, np.int64),
             paths=np.asarray([str(p) for p in paths]))
    print(f"[clip:{split}] cached -> {out}")
    return out


def load(split: str) -> tuple[np.ndarray, np.ndarray]:
    """(emb [N,512], concept [N]) from cache — computes it first if absent."""
    d = np.load(compute(split))
    return d["emb"], d["concept"]


def embeddings_by_file(split: str) -> dict[str, np.ndarray]:
    """{image basename -> CLIP embedding} for a split — lets the training runner align each EEG epoch to the
    exact image it viewed (the adapter returns the basename)."""
    d = np.load(compute(split))
    names = [Path(str(p)).name for p in d["paths"]]
    return dict(zip(names, d["emb"]))


def concept_prototypes(split: str) -> np.ndarray:
    """Mean CLIP embedding per concept (re-normalized) -> [n_concepts, 512]; the retrieval candidate bank."""
    emb, concept = load(split)
    n = int(concept.max()) + 1
    proto = np.zeros((n, 512), np.float32)
    for c in range(n):
        v = emb[concept == c].mean(0)
        proto[c] = v / (np.linalg.norm(v) + 1e-8)
    return proto


if __name__ == "__main__":
    import sys

    compute(sys.argv[1] if len(sys.argv) > 1 else "test")
