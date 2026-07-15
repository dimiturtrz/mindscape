"""Equivalence-class tests for the Normalizer interface + CompositeNormalization composite (bd cx7x).

Partition: (1) an empty chain is the identity; (2) a chain applies its links in order; (3) the chain is itself
a Normalizer (composes). A tiny scalar-add link stands in for a concrete normalizer."""
import numpy as np

from core.normalization.normalization import CompositeNormalization, NormContext, Normalizer


class _AddK(Normalizer):
    """Test link: add a constant (order-sensitive when composed with a scale)."""

    def __init__(self, k):
        self.k = k

    def apply(self, X, ctx):
        return X + self.k


class _MulK(Normalizer):
    def __init__(self, k):
        self.k = k

    def apply(self, X, ctx):
        return X * self.k


def test_empty_chain_is_identity():
    X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    out = CompositeNormalization([]).apply(X, NormContext())
    np.testing.assert_array_equal(out, X)


def test_chain_applies_links_in_order():
    """(x + 1) * 2 ≠ (x * 2) + 1 — order is respected."""
    X = np.zeros((1, 1, 1), dtype=np.float32)
    add_then_mul = CompositeNormalization([_AddK(1.0), _MulK(2.0)]).apply(X, NormContext())
    mul_then_add = CompositeNormalization([_MulK(2.0), _AddK(1.0)]).apply(X, NormContext())
    assert add_then_mul.item() == 2.0
    assert mul_then_add.item() == 1.0


def test_chain_is_a_normalizer():
    """A chain nests inside another chain (composite)."""
    X = np.zeros((1, 1, 1), dtype=np.float32)
    inner = CompositeNormalization([_AddK(3.0)])
    outer = CompositeNormalization([inner, _MulK(2.0)])
    assert isinstance(inner, Normalizer)
    assert outer.apply(X, NormContext()).item() == 6.0
