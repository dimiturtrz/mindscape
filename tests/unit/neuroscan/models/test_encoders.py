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


def test_nice_builds_and_honours_the_contract():
    spec = EncoderSpec(n_channels=17, n_times=100, embed_dim=64)
    encoder = EncoderRegistry.build_encoder("nice", spec)
    z = encoder(torch.randn(4, spec.n_channels, spec.n_times))
    assert z.shape == (4, spec.embed_dim)                       # [B, embed_dim]
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)   # L2-normalized in CLIP space


def test_unknown_encoder_raises_with_known_names():
    with pytest.raises(KeyError, match="nice"):                 # message lists the available names
        EncoderRegistry.build_encoder("nope", EncoderSpec(n_channels=17, n_times=100, embed_dim=64))


@pytest.mark.parametrize(("override", "link"), [
    ("auto", ZScore), ("zscore", ZScore), ("scale", Scale), ("mvnn", Mvnn)])
def test_normalization_override_resolves_to_its_link(override, link):
    """Each --normalize override maps to the right normalizer object: auto/zscore → per-channel z-score (the
    default for every encoder), scale → CBraMod amplitude scale, mvnn → per-subject whitening."""
    groups = np.zeros(4, dtype=np.int64)
    conditions = np.repeat(np.arange(2), 2)
    chain = EncoderRegistry.normalization("nice", override, groups, conditions)
    assert len(chain.links) == 1 and isinstance(chain.links[0], link)
