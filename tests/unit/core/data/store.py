"""store.gather — pulls epochs for a split frame in row order, reading each subject npz once."""
import numpy as np
import polars as pl

from core.data import store


def test_gather(tmp_path):
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


def test_gather_aligned(tmp_path):
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


class _FakeAdapter:
    """A minimal in-memory DatasetAdapter: 2 subjects × 1 epoch, so build() has real arrays to cache/frame."""
    label_map = {"class_a": 0, "class_b": 1}

    def subjects(self):
        return [1, 2]

    def get_data(self, subs, cfg):
        n = len(subs)
        return (np.zeros((n, 2, 10), dtype=np.float32), np.array([0] * n),
                pl.DataFrame({"session": ["s1"] * n, "run": ["0"] * n}))


def test_build(tmp_path, monkeypatch):
    """build() resolves the adapter, epochs every subject, and writes the per-dataset cache + meta index."""
    from core.data.eeg.base import EpochCfg
    monkeypatch.setenv("MINDSCAPE_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(store.Registry, "get_adapter", lambda name: _FakeAdapter())

    out = store.Store.build("test_dataset", EpochCfg())
    assert (out / "meta.csv").exists()
    meta = pl.read_csv(out / "meta.csv")
    assert [str(s) for s in meta["subject"].to_list()] == ["1", "2"]   # one epoch per fake subject, in order
    assert set(meta["label"].to_list()) == {"class_a"}                 # y=0 -> the id->name mapping applied


def test_dataset_dir(tmp_path, monkeypatch):
    """Test that dataset_dir returns correct path."""
    from core.data.eeg.base import EpochCfg
    monkeypatch.setenv("MINDSCAPE_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    cfg = EpochCfg()
    dataset_dir = store.Store.dataset_dir("test", cfg)
    assert "test" in str(dataset_dir)
    assert cfg.key() in str(dataset_dir)


def test_load(tmp_path, monkeypatch):
    """Test that load returns a polars DataFrame."""
    from core.data.eeg.base import EpochCfg

    monkeypatch.setenv("MINDSCAPE_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()

    cfg = EpochCfg()
    out_dir = store.Store.dataset_dir("test", cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup a minimal metadata file
    meta_file = out_dir / "meta.csv"
    meta_data = pl.DataFrame({
        "dataset": ["test"], "subject": ["1"], "session": ["s1"], "run": ["0"],
        "label_id": [0], "label": ["left_hand"], "epoch": [0], "file": ["sub1.npz"]
    })
    meta_data.write_csv(meta_file)

    df = store.Store.load("test", cfg)
    assert isinstance(df, pl.DataFrame)


def test_channels(tmp_path, monkeypatch):
    """Test that channels returns channel names or None."""
    from core.data.eeg.base import EpochCfg
    import json

    monkeypatch.setenv("MINDSCAPE_DATA", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()

    cfg = EpochCfg()
    out_dir = store.Store.dataset_dir("test", cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create metadata
    meta_file = out_dir / "meta.csv"
    meta_data = pl.DataFrame({
        "dataset": ["test"], "subject": ["1"], "session": ["s1"], "run": ["0"],
        "label_id": [0], "label": ["left_hand"], "epoch": [0], "file": ["sub1.npz"]
    })
    meta_data.write_csv(meta_file)

    # Create channels file
    channels_file = out_dir / "channels.json"
    channels_file.write_text(json.dumps(["ch1", "ch2", "ch3"]))

    channels = store.Store.channels("test", cfg)
    assert channels == ["ch1", "ch2", "ch3"]
