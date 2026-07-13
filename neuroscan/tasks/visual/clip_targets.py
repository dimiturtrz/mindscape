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

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from core.config import Config
from neuroscan.tasks.cli import Cli

logger = logging.getLogger(__name__)

_ROOT = "things_eeg2"
_CLIP_ARCH, _CLIP_PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"   # open_clip model (NICE uses a CLIP ViT)
_EMBED_DIM = 512
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class ImageItem:
    """One stimulus image and the concept it belongs to (concept index = sorted-folder position)."""
    path: Path
    concept: int


class ClipTargets:
    """CLIP image-embedding targets for the retrieval baseline — the free helpers folded in as staticmethods
    (public names kept). Walks the THINGS-EEG2 image set, caches per-image embeddings + concept index, and
    serves them as prototypes / per-file lookups for the trainer."""

    @staticmethod
    def _image_root(split: str) -> Path:
        return Config.raw_dir() / _ROOT / "images" / f"{split}_images"

    @staticmethod
    def concept_dirs(split: str) -> list[Path]:
        """Sorted concept folders for a split (the sort defines the concept index used everywhere)."""
        return sorted(path for path in ClipTargets._image_root(split).iterdir() if path.is_dir())

    @staticmethod
    def _list_images(split: str) -> list[ImageItem]:
        """All stimulus images of a split as (path, concept) items, concept-sorted then filename-sorted.

        Two levels because the layout is genuinely two levels (concept folder -> its exemplar images); each item
        carries its own concept, so downstream code never re-derives it from a parallel array.
        """
        return [
            ImageItem(image_path, concept)
            for concept, concept_dir in enumerate(ClipTargets.concept_dirs(split))
            for image_path in sorted(concept_dir.iterdir())
            if image_path.suffix.lower() in _IMAGE_EXTS
        ]

    @staticmethod
    def _load_clip(device: str):
        model, _, preprocess = open_clip.create_model_and_transforms(_CLIP_ARCH, pretrained=_CLIP_PRETRAINED)
        return model.eval().to(device), preprocess

    @staticmethod
    def compute(split: str, *, device: str | None = None, batch: int = 256, force: bool = False) -> Path:
        """Embed every image in `split` with CLIP, cache to `<processed>/things_eeg2/clip_<split>.npz`.

        npz: `emb` [N, 512] float32 (L2-normalized), `concept` [N] int (concept index), `paths` [N] str.
        Idempotent — returns the cache path, recomputing only if missing or `force`.
        """
        cache_path = Config.processed_dir() / _ROOT / f"clip_{split}.npz"
        if cache_path.exists() and not force:
            return cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, preprocess = ClipTargets._load_clip(device)

        items = ClipTargets._list_images(split)
        embeddings = np.empty((len(items), _EMBED_DIM), dtype=np.float32)
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            pixels = torch.stack([preprocess(Image.open(item.path).convert("RGB")) for item in chunk]).to(device)
            with torch.no_grad():
                encoded = model.encode_image(pixels).float()
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            embeddings[start:start + len(chunk)] = encoded.cpu().numpy()
            logger.info(f"[clip:{split}] {start + len(chunk)}/{len(items)}")
        np.savez(cache_path,
                 emb=embeddings,
                 concept=np.asarray([item.concept for item in items], np.int64),
                 paths=np.asarray([str(item.path) for item in items]))
        logger.info(f"[clip:{split}] cached -> {cache_path}")
        return cache_path

    @staticmethod
    def load(split: str) -> tuple[np.ndarray, np.ndarray]:
        """(emb [N,512], concept [N]) from cache — computes it first if absent."""
        cached = np.load(ClipTargets.compute(split))
        return cached["emb"], cached["concept"]

    @staticmethod
    def embeddings_by_file(split: str) -> dict[str, np.ndarray]:
        """{image basename -> CLIP embedding} for a split — lets the training runner align each EEG epoch to the
        exact image it viewed (the adapter returns the basename)."""
        cached = np.load(ClipTargets.compute(split))
        basenames = [Path(str(path)).name for path in cached["paths"]]
        return dict(zip(basenames, cached["emb"], strict=True))

    @staticmethod
    def concept_prototypes(split: str) -> np.ndarray:
        """Mean CLIP embedding per concept (re-normalized) -> [n_concepts, 512]; the retrieval candidate bank."""
        embeddings, concept = ClipTargets.load(split)
        n_concepts = int(concept.max()) + 1
        prototypes = np.zeros((n_concepts, _EMBED_DIM), np.float32)
        for concept_idx in range(n_concepts):
            mean_emb = embeddings[concept == concept_idx].mean(0)
            prototypes[concept_idx] = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        return prototypes


def main():
    Cli.setup_logging()
    ClipTargets.compute(sys.argv[1] if len(sys.argv) > 1 else "test")


if __name__ == "__main__":
    main()
