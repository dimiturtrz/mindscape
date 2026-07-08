"""NICE encoder + contrastive/retrieval primitives — data-free contracts."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from neuroscan.models.nice import NiceConfig, NiceEncoder, clip_infonce, retrieval_topk


def _pair(seed=0, b=8, d=32):
    g = torch.Generator().manual_seed(seed)
    eeg = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    img = F.normalize(torch.randn(b, d, generator=g), dim=-1)
    return eeg, img, torch.tensor(14.0)


def test_clip_infonce_hard_beta_zero_is_exact_standard():
    eeg, img, scale = _pair()
    logits = scale * eeg @ img.t()
    target = torch.arange(8)
    std = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
    assert torch.allclose(clip_infonce(eeg, img, scale, hard_beta=0.0), std)   # bd fww: off by default


def test_clip_infonce_hard_beta_raises_loss_on_hard_negatives():
    eeg, img, scale = _pair()
    base = clip_infonce(eeg, img, scale, hard_beta=0.0)
    hard = clip_infonce(eeg, img, scale, hard_beta=1.0)
    assert hard > base                                        # boosting hard negatives increases the loss


def test_encoder_shape_and_norm():
    enc = NiceEncoder(NiceConfig(n_channels=63, n_times=250, embed_dim=512)).eval()
    z = enc(torch.randn(8, 63, 250))
    assert z.shape == (8, 512)
    assert torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-4)   # L2-normalized


def test_infonce_rewards_matches():
    """Matched EEG==image embeddings give near-zero loss; shuffled targets give a larger loss."""
    torch.manual_seed(0)
    z = torch.nn.functional.normalize(torch.randn(16, 512), dim=-1)
    ls = torch.tensor(20.0)
    matched = clip_infonce(z, z, ls)
    shuffled = clip_infonce(z, z[torch.randperm(16)], ls)
    assert matched < 0.1
    assert shuffled > matched


def test_retrieval_topk_planted_and_chance():
    cand = torch.nn.functional.normalize(torch.randn(200, 512), dim=-1)
    labels = torch.arange(200)
    perfect = retrieval_topk(cand.clone(), cand, labels)          # each queries its own candidate
    assert perfect[1] == 1.0 and perfect[5] == 1.0
    rand = torch.nn.functional.normalize(torch.randn(200, 512), dim=-1)
    chance = retrieval_topk(rand, cand, labels)                   # unrelated queries ~ chance
    assert chance[1] < 0.1                                        # 1/200 = 0.5%, well under 10%
