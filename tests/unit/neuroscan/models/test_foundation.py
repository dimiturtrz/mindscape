"""CBraMod foundation encoder (bd yjd) — the pretrained backbone behind the EEG→CLIP contract.

Integration test: needs the checked-out backbone (external/CBraMod) + out-of-repo pretrained weights, so it
SKIPS where those are absent (CI, a fresh clone). Equivalence classes: the encoder honours the contract
(L2-normalized [B, embed_dim]) and the frozen backbone contributes no trainable params.
"""
import pytest
import torch

from core.config import Config
from neuroscan.models import foundation
from neuroscan.models.encoders import EncoderSpec, build_encoder

_CKPT = Config.data_root("pretrained") / "CBraMod" / "pretrained_weights.pth"
_AVAILABLE = foundation._CBRAMOD_ROOT.exists() and _CKPT.exists()
pytestmark = pytest.mark.skipif(not _AVAILABLE, reason="CBraMod checkout + weights not present (out-of-repo)")


def test_cbramod_honours_contract_with_frozen_backbone():
    spec = EncoderSpec(n_channels=63, n_times=200, embed_dim=1024)   # 1 s @ 200 Hz, CLIP dim
    enc = build_encoder("cbramod", spec)
    z = enc(torch.randn(3, spec.n_channels, spec.n_times))
    assert z.shape == (3, spec.embed_dim)
    assert torch.allclose(z.norm(dim=-1), torch.ones(3), atol=1e-5)   # L2-normalized in CLIP space
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("head") for n in trainable)  # only the head learns; backbone frozen


def test_frozen_backbone_stays_eval_when_trained():
    enc = build_encoder("cbramod", EncoderSpec(n_channels=63, n_times=200, embed_dim=1024))
    enc.train()
    assert not enc.backbone.training       # frozen backbone kept in eval (no dropout/norm drift)
