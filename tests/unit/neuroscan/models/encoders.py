"""Encoder registry (bd bji) — name -> a builder honouring the EEG→CLIP contract.

Equivalence classes: a registered name builds an encoder with the right output contract (L2-normalized
[B, embed_dim]), and an unknown name is a clean KeyError listing what's available.
"""
import numpy as np
import pytest
import torch

from core.normalization.mvnn import Mvnn
from core.normalization.scale import Scale
from core.normalization.zscore import ZScore
from neuroscan.models.encoders import EncoderRegistry, EncoderSpec

torch.manual_seed(0)


def test_build_encoder():
    spec = EncoderSpec(n_channels=17, n_times=100, embed_dim=64)
    encoder = EncoderRegistry.build_encoder("nice", spec)
    z = encoder(torch.randn(4, spec.n_channels, spec.n_times))
    assert z.shape == (4, spec.embed_dim)                       # [B, embed_dim]
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)   # L2-normalized in CLIP space


def test_unknown_encoder_raises_with_known_names():
    with pytest.raises(KeyError, match="nice"):                 # message lists the available names
        EncoderRegistry.build_encoder("nope", EncoderSpec(n_channels=17, n_times=100, embed_dim=64))


def test_register():
    """EncoderRegistry.register adds an encoder to the registry under a name."""
    # Create a simple test encoder builder
    def test_builder(spec):
        return torch.nn.Identity()

    # Register it, then build under that name — the built object must be what THIS builder returns (an
    # Identity), proving register wired the name to the builder (not just that some encoder exists).
    EncoderRegistry.register("test_encoder_temp", test_builder)
    spec = EncoderSpec(n_channels=8, n_times=64, embed_dim=128)
    encoder = EncoderRegistry.build_encoder("test_encoder_temp", spec)
    assert isinstance(encoder, torch.nn.Identity)


@pytest.mark.parametrize(("override", "link"), [
    ("auto", ZScore), ("zscore", ZScore), ("scale", Scale), ("mvnn", Mvnn)])
def test_normalization(override, link):
    """Each --normalize override maps to the right normalizer object: auto/zscore → per-channel z-score (the
    default for every encoder), scale → CBraMod amplitude scale, mvnn → per-subject whitening."""
    groups = np.zeros(4, dtype=np.int64)
    conditions = np.repeat(np.arange(2), 2)
    chain = EncoderRegistry.normalization("nice", override, groups, conditions)
    assert len(chain.links) == 1 and isinstance(chain.links[0], link)
