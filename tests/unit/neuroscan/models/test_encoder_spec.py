"""EEG->CLIP encoder contract (`neuroscan.models.encoder_spec`).

`EncoderSpec` is the data-derived shape (pydantic): it must store the three fields, coerce clean int-like
input, and reject a non-coercible type with a ValidationError. `ImageEncoder` is the forward Protocol — a
class with a matching `forward` conforms structurally.
"""
import pytest
from pydantic import ValidationError

from neuroscan.models.encoder_spec import EncoderSpec, ImageEncoder


def test_encoder_spec_stores_the_three_shape_fields():
    spec = EncoderSpec(n_channels=64, n_times=200, embed_dim=512)
    assert (spec.n_channels, spec.n_times, spec.embed_dim) == (64, 200, 512)


def test_encoder_spec_rejects_non_coercible_type():
    with pytest.raises(ValidationError):
        EncoderSpec(n_channels="sixty-four", n_times=200, embed_dim=512)


def test_encoder_spec_requires_all_fields():
    with pytest.raises(ValidationError):
        EncoderSpec(n_channels=64, n_times=200)          # embed_dim missing


def test_conforming_class_satisfies_image_encoder_protocol():
    """A class exposing `forward` conforms to the ImageEncoder contract the trainer programs against."""
    class Dummy:
        def forward(self, x):
            return x

    enc: ImageEncoder = Dummy()
    assert hasattr(enc, "forward")
    assert enc.forward(7) == 7
