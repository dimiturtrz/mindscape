"""Encoder registry (bd bji) — name -> a builder honouring the EEG→CLIP contract.

Equivalence classes: a registered name builds an encoder with the right output contract (L2-normalized
[B, embed_dim]), and an unknown name is a clean KeyError listing what's available.
"""
import pytest
import torch

from neuroscan.models.encoders import EncoderSpec, build_encoder


def test_nice_builds_and_honours_the_contract():
    spec = EncoderSpec(n_channels=17, n_times=100, embed_dim=64)
    encoder = build_encoder("nice", spec)
    z = encoder(torch.randn(4, spec.n_channels, spec.n_times))
    assert z.shape == (4, spec.embed_dim)                       # [B, embed_dim]
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)   # L2-normalized in CLIP space


def test_unknown_encoder_raises_with_known_names():
    with pytest.raises(KeyError, match="nice"):                 # message lists the available names
        build_encoder("nope", EncoderSpec(n_channels=17, n_times=100, embed_dim=64))
