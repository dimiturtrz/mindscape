"""store.gather — pulls epochs for a split frame in row order, reading each subject npz once."""
import numpy as np
import polars as pl

from core.data import store


def test_gather_preserves_row_order(tmp_path):
    X = np.arange(5 * 2 * 3).reshape(5, 2, 3).astype(np.float32)
    npz = tmp_path / "sub1.npz"
    np.savez(npz, X=X, y=np.array([0, 1, 2, 3, 0]),
             session=np.array(["s"] * 5), run=np.array(["0"] * 5))
    # request epochs 4, 0, 2 (out of natural order) — gather must return them in THIS order
    df = pl.DataFrame({"path": [str(npz)] * 3, "epoch": [4, 0, 2], "label_id": [0, 0, 2]})
    Xo, yo = store.Store.gather(df)
    assert Xo.shape == (3, 2, 3)
    assert np.array_equal(Xo[0], X[4])
    assert np.array_equal(Xo[1], X[0])
    assert np.array_equal(Xo[2], X[2])
    assert list(yo) == [0, 0, 2]


def test_gather_spans_multiple_subject_files(tmp_path):
    Xa = np.zeros((2, 1, 2), np.float32)
    Xb = np.ones((2, 1, 2), np.float32)
    np.savez(tmp_path / "a.npz", X=Xa, y=np.array([0, 0]))
    np.savez(tmp_path / "b.npz", X=Xb, y=np.array([1, 1]))
    df = pl.DataFrame({
        "path": [str(tmp_path / "a.npz"), str(tmp_path / "b.npz"), str(tmp_path / "a.npz")],
        "epoch": [0, 1, 1], "label_id": [0, 1, 0]})
    Xo, yo = store.Store.gather(df)
    assert Xo[0].sum() == 0 and Xo[1].sum() == 2 and Xo[2].sum() == 0
    assert list(yo) == [0, 1, 0]


def test_gather_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        store.Store.gather(pl.DataFrame({"path": [], "epoch": [], "label_id": []}))


def _meta(npz, subject, labels):
    return pl.DataFrame({
        "path": [str(npz)] * len(labels), "epoch": list(range(len(labels))),
        "label_id": list(labels), "subject": [str(subject)] * len(labels)})


def test_gather_aligned_returns_both_modalities_and_shared_labels(tmp_path):
    Xe = np.arange(3 * 2 * 4).reshape(3, 2, 4).astype(np.float32)
    Xf = np.arange(3 * 5 * 6).reshape(3, 5, 6).astype(np.float32)     # fNIRS: different ch/t
    np.savez(tmp_path / "e.npz", X=Xe, y=np.array([0, 1, 0]))
    np.savez(tmp_path / "f.npz", X=Xf, y=np.array([0, 1, 0]))
    me = _meta(tmp_path / "e.npz", 1, [0, 1, 0])
    mf = _meta(tmp_path / "f.npz", 1, [0, 1, 0])
    xe, xf, y = store.Store.gather_aligned(me, mf, 1)                 # int subject coerced to str
    assert xe.shape == (3, 2, 4) and xf.shape == (3, 5, 6)
    assert list(y) == [0, 1, 0]


def test_gather_aligned_raises_on_misaligned_labels(tmp_path):
    import pytest
    np.savez(tmp_path / "e.npz", X=np.zeros((2, 1, 1), np.float32), y=np.array([0, 1]))
    np.savez(tmp_path / "f.npz", X=np.zeros((2, 1, 1), np.float32), y=np.array([1, 0]))
    me = _meta(tmp_path / "e.npz", "7", [0, 1])
    mf = _meta(tmp_path / "f.npz", "7", [1, 0])
    with pytest.raises(ValueError, match="misaligned"):
        store.Store.gather_aligned(me, mf, "7")
