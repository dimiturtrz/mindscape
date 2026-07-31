"""Equivalence-class tests for the Normalizer interface + CompositeNormalization composite (bd cx7x).

Partition: (1) an empty chain is the identity; (2) a chain applies its links in order; (3) the chain is itself
a Normalizer (composes); (4) chain.fit fits every link (each on the running output of the ones before it)."""
import numpy as np

from core.normalization.normalization import CompositeNormalization, Normalizer


class _MulK(Normalizer):
    """Stateless test link: multiply by a constant."""

    def __init__(self, k):
        self.k = k

    def apply(self, X, groups=None):
        return X * self.k


class _AddK(Normalizer):
    """Test link that must be FIT before it adds anything — proves the composite fits each link."""

    def __init__(self, k):
        self.k = k
        self._ready = 0.0

    def fit(self, X):
        self._ready = self.k
        return self

    def apply(self, X, groups=None):
        return X + self._ready


def test_empty_chain_is_identity():
    X = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.testing.assert_array_equal(CompositeNormalization([]).apply(X), X)


def test_chain_applies_links_in_order():
    """(x + 1) * 2 ≠ (x * 2) + 1 — order is respected (add-links fit first)."""
    X = np.zeros((1, 1, 1), dtype=np.float32)
    add_then_mul = CompositeNormalization([_AddK(1.0), _MulK(2.0)]).fit(X).apply(X)
    mul_then_add = CompositeNormalization([_MulK(2.0), _AddK(1.0)]).fit(X).apply(X)
    assert add_then_mul.item() == 2.0
    assert mul_then_add.item() == 1.0


def test_chain_is_a_normalizer():
    """A chain nests inside another chain (composite)."""
    X = np.zeros((1, 1, 1), dtype=np.float32)
    inner = CompositeNormalization([_AddK(3.0)])
    outer = CompositeNormalization([inner, _MulK(2.0)])
    assert isinstance(inner, Normalizer)
    assert outer.fit(X).apply(X).item() == 6.0


def test_chain_fit_fits_every_link():
    """Without fit, an _AddK contributes 0; chain.fit must arm each link (else apply ≠ expected)."""
    X = np.zeros((1, 1, 1), dtype=np.float32)
    chain = CompositeNormalization([_AddK(5.0), _AddK(2.0)])
    assert chain.apply(X).item() == 0.0          # unfit: both add 0
    assert chain.fit(X).apply(X).item() == 7.0   # fit: 5 + 2
