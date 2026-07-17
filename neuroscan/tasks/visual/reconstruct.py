"""EEG -> image reconstruction (bd 71n): decode a predicted CLIP ViT-L/14 embedding into an image.

Endpoint of the Stage-3 arc — the retrieval number was always the proxy; this is the payoff. A trained encoder
(saved by train_nice --config perception_ft_recon.json, save_encoder=True) predicts a CLIP ViT-L/14 image
embedding from EEG; StableUnCLIP (stabilityai/stable-diffusion-2-1-unclip-small, OpenAI ViT-L/14 space, 768-d,
`image_embeds` injection) decodes it to an image.

Two modes, so decoder and EEG quality read SEPARATELY (decoder recipe / CLIP-space caveats in the deep-dive
research/deep_dives/2026-07-17_clip_embedding_decoder_options.md):
  sanity -- decode the TRUE test-concept CLIP embeddings (no EEG, no encoder): the decoder CEILING.
  eeg    -- decode the held-out subject's EEG-PREDICTED per-concept embeddings: the actual result.

Output: a per-concept triptych row [true stimulus | decode(true CLIP) | decode(EEG-pred)] tiled into one PNG.
Coverage-omitted shell: diffusers + ~5 GB model download + device; the decodable objective/checkpoint it
consumes is the tested logic (train_nice, bd ooi).
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int
from PIL import Image

from neuroscan.models.encoders import EncoderRegistry, EncoderSpec
from neuroscan.tasks.cli import Cli
from neuroscan.tasks.visual import clip_targets
from neuroscan.tasks.visual.train_nice import TrainConfig, TrainNice

logger = logging.getLogger(__name__)

# img2img, ViT-L/14 (image_encoder projection_dim 768, OpenAI CLIP); -i2i-h is ViT-H/14 (1024) = wrong dim
_UNCLIP = "diffusers/stable-diffusion-2-1-unclip-i2i-l"
_TILE = 224            # per-image square in the output grid
_EVAL_BATCH = 256
_DEFAULT_TARGET = "vitl14"


@dataclass
class ReconConfig:
    """One reconstruction run (config-object, not a param list)."""
    mode: str = "sanity"
    encoder_ckpt: str | None = None
    test_subject: int = 5
    n: int = 8
    steps: int = 20
    noise_level: int = 0
    out_path: Path = Path("runs/reconstruct.png")


class Reconstruct:
    """EEG->image via unCLIP decode of a predicted CLIP embedding — op-namespace of staticmethods (bd 71n)."""

    @staticmethod
    def _pipe(device: str):
        """StableUnCLIP image-variation pipeline, fp16 on cuda. Lazy import: diffusers is the `recon` extra."""
        from diffusers import StableUnCLIPImg2ImgPipeline  # noqa: PLC0415 — heavy optional dep, load on use
        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(_UNCLIP, torch_dtype=dtype)
        return pipe.to(device)

    @staticmethod
    def decode(pipe, embeds: Float[np.ndarray, "n d"], device: str,
               steps: int = 20, noise_level: int = 0) -> list[Image.Image]:
        """Decode CLIP image embeddings [n, 768] -> PIL images via `image_embeds` injection (skips the encoder)."""
        dtype = torch.float16 if device == "cuda" else torch.float32
        tensor = torch.from_numpy(np.ascontiguousarray(embeds)).to(device=device, dtype=dtype)
        images = []
        for i in range(len(tensor)):
            out = pipe(image_embeds=tensor[i:i + 1], prompt="", num_inference_steps=steps, noise_level=noise_level)
            images.append(out.images[0])
        return images

    @staticmethod
    def _load_encoder(ckpt_path: Path, device: str):
        """Rebuild the encoder from its saved spec and load the trained weights (train_nice save_encoder)."""
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        spec = EncoderSpec(n_channels=ckpt["n_channels"], n_times=ckpt["n_times"], embed_dim=ckpt["embed_dim"])
        encoder = EncoderRegistry.build_encoder(ckpt["model"], spec).to(device)
        encoder.load_state_dict(ckpt["state_dict"])
        return encoder.eval(), ckpt

    @staticmethod
    def _predict(encoder, eeg: Float[np.ndarray, "n ch t"], device: str) -> Float[np.ndarray, "n d"]:
        """EEG [n, ch, t] -> L2-normalized CLIP-space embeddings [n, d] (the encoder's forward, batched)."""
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            chunks = [encoder(torch.from_numpy(eeg[i:i + _EVAL_BATCH]).to(device)).float().cpu()
                      for i in range(0, len(eeg), _EVAL_BATCH)]
        return torch.cat(chunks).numpy()

    @staticmethod
    def _concept_mean(embeds: Float[np.ndarray, "n d"], concept: Int[np.ndarray, "n"],
                      n_concepts: int) -> Float[np.ndarray, "k d"]:
        """Per-concept mean embedding, re-normalized [n_concepts, d] — denoises the single-trial prediction."""
        out = np.zeros((n_concepts, embeds.shape[1]), np.float32)
        for c in range(n_concepts):
            mean = embeds[concept == c].mean(0)
            out[c] = mean / (np.linalg.norm(mean) + 1e-8)
        return out

    @staticmethod
    def _stimulus_images(concepts: list[int]) -> list[Image.Image]:
        """The first stimulus image of each requested test concept (the ground-truth the subject viewed)."""
        dirs = clip_targets.ClipTargets.concept_dirs("test")
        images = []
        for c in concepts:
            files = sorted(p for p in dirs[c].iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            images.append(Image.open(files[0]).convert("RGB"))
        return images

    @staticmethod
    def _grid(rows: list[list[Image.Image]], out_path: Path, labels: tuple[str, ...]) -> None:
        """Tile rows of images into one PNG; each row is one concept, columns share the `labels` header."""
        n_cols = max(len(r) for r in rows)
        header = 20
        canvas = Image.new("RGB", (n_cols * _TILE, header + len(rows) * _TILE), "white")
        for r, row in enumerate(rows):
            for c, img in enumerate(row):
                canvas.paste(img.resize((_TILE, _TILE)), (c * _TILE, header + r * _TILE))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)
        logger.info(f"saved {len(rows)}x{n_cols} grid ({', '.join(labels)}) -> {out_path}")

    @staticmethod
    def run(cfg: ReconConfig) -> None:
        """Reconstruct `cfg.n` test concepts. sanity = decoder ceiling (true CLIP); eeg = the EEG-predicted result."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        target = _DEFAULT_TARGET
        pipe = Reconstruct._pipe(device)

        pred = None
        if cfg.mode == "eeg":
            encoder, ckpt = Reconstruct._load_encoder(Path(cfg.encoder_ckpt), device)
            target = ckpt.get("clip_target", _DEFAULT_TARGET)
            tcfg = TrainConfig(model=ckpt["model"], clip_target=target, resample=ckpt.get("resample", 200))
            test_eeg, test_concept = TrainNice.test_features(cfg.test_subject, tcfg)
            pred = Reconstruct._concept_mean(
                Reconstruct._predict(encoder, test_eeg, device), test_concept, int(test_concept.max()) + 1)

        true_emb = clip_targets.ClipTargets.concept_prototypes("test", target)
        concepts = list(range(min(cfg.n, len(true_emb))))
        stimuli = Reconstruct._stimulus_images(concepts)
        decoded_true = Reconstruct.decode(pipe, true_emb[concepts], device, cfg.steps, cfg.noise_level)

        if cfg.mode == "eeg":
            decoded_eeg = Reconstruct.decode(pipe, pred[concepts], device, cfg.steps, cfg.noise_level)
            rows = [[stimuli[i], decoded_true[i], decoded_eeg[i]] for i in range(len(concepts))]
            labels = ("stimulus", "decode(true CLIP)", "decode(EEG)")
        else:
            rows = [[stimuli[i], decoded_true[i]] for i in range(len(concepts))]
            labels = ("stimulus", "decode(true CLIP)")
        Reconstruct._grid(rows, cfg.out_path, labels)


def main():
    Cli.setup_logging()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["sanity", "eeg"], default="sanity",
                    help="sanity = decode true CLIP embeds (decoder ceiling); eeg = decode EEG-predicted embeds")
    ap.add_argument("--encoder", default=None, help="encoder checkpoint (runs/enc_*.pt) — required for --mode eeg")
    ap.add_argument("--test", type=int, default=5, help="held-out test subject id")
    ap.add_argument("--n", type=int, default=8, help="number of test concepts to reconstruct")
    ap.add_argument("--steps", type=int, default=20, help="diffusion inference steps")
    ap.add_argument("--noise-level", dest="noise_level", type=int, default=0, help="unCLIP noise level (0..1000)")
    ap.add_argument("--out", default="runs/reconstruct.png", help="output grid PNG")
    args = ap.parse_args()
    Reconstruct.run(ReconConfig(mode=args.mode, encoder_ckpt=args.encoder, test_subject=args.test, n=args.n,
                                steps=args.steps, noise_level=args.noise_level, out_path=Path(args.out)))


if __name__ == "__main__":
    main()
