"""NICE encoder — the EEG-to-CLIP feature map."""
from __future__ import annotations

import pytest
import torch

from core.features.eeg.montage import EegMontage
from neuroscan.models.nice import NiceConfig, NiceEncoder

torch.manual_seed(0)


def test_forward():
    enc = NiceEncoder(NiceConfig(n_channels=63, n_times=250, embed_dim=512)).eval()
    with torch.no_grad():
        z = enc.forward(torch.randn(8, 63, 250))
    assert z.shape == (8, 512)
    assert torch.allclose(z.norm(dim=-1), torch.ones(8), atol=1e-4)   # L2-normalized


def test_geo_penalty():
    """Graph-Laplacian smoothness penalty (bd 1x0): zero when every channel's spatial weights are equal (a
    perfectly smooth map pays nothing), strictly positive once neighbours differ."""
    enc = NiceEncoder(NiceConfig(n_channels=6, n_times=250, embed_dim=32))
    lap = torch.tensor(EegMontage.channel_laplacian(
        EegMontage.eeg_positions(["Cz", "Fz", "Pz", "Oz", "C3", "C4"]), sigma=0.3))
    with torch.no_grad():                                       # make every channel column identical -> smooth
        enc.spatial.weight.copy_(enc.spatial.weight[:, :, :1, :].expand_as(enc.spatial.weight).contiguous())
    assert enc.geo_penalty(lap).item() == pytest.approx(0.0, abs=1e-5)
    with torch.no_grad():
        enc.spatial.weight.random_()                            # arbitrary per-channel weights -> pays a cost
    assert enc.geo_penalty(lap).item() > 0.0
