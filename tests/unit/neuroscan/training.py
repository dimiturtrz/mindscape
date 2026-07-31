"""Unit tests for the shared training scaffold (bd 1eca) — EarlyStopper direction/patience/restore."""
import torch

from neuroscan.training import EarlyStopper


def _lin(bias: float) -> torch.nn.Module:
    m = torch.nn.Linear(1, 1)
    with torch.no_grad():
        m.bias.fill_(bias)
    return m


def test_stopper_max_tracks_best_and_restores():
    """mode='max' (val top-1): keeps the highest-metric checkpoint, restores it over a later worse one."""
    s = EarlyStopper(patience=0, mode="max")
    assert s.update(0.10, _lin(1.0), step=0) is False
    assert s.update(0.30, _lin(2.0), step=1) is False       # improves -> new best
    assert s.update(0.20, _lin(3.0), step=2) is False       # worse -> best unchanged
    assert s.best_step == 1
    assert s.best_metric == 0.30
    m = _lin(9.0)
    s.restore(m)
    assert m.bias.item() == 2.0                             # restored the step-1 weights, not the last


def test_update():
    """EarlyStopper.update returns False while training should continue, True once `patience` bad epochs pass
    (stop when bad >= patience)."""
    s = EarlyStopper(patience=2, mode="max")
    assert s.update(0.5, _lin(1.0), step=0) is False        # first -> new best, bad=0
    assert s.update(0.4, _lin(2.0), step=1) is False        # 1st bad, bad=1 < patience=2 -> continue
    assert s.update(0.3, _lin(3.0), step=2) is True         # 2nd bad, bad=2 >= patience -> stop


def test_best_metric():
    """EarlyStopper.best_metric tracks the best value seen."""
    s = EarlyStopper(patience=2, mode="min", min_delta=1e-4)
    s.update(1.0, _lin(0.0))
    assert s.best_metric == 1.0
    s.update(0.9999, _lin(0.0))
    assert s.best_metric == 1.0  # within min_delta, not an improvement
    s.update(0.5, _lin(0.0))
    assert s.best_metric == 0.5  # real improvement


def test_restore():
    """EarlyStopper.restore restores the best checkpoint."""
    s = EarlyStopper(patience=0, mode="max")
    s.update(0.10, _lin(1.0), step=0)
    s.update(0.30, _lin(2.0), step=1)
    s.update(0.20, _lin(3.0), step=2)
    m = _lin(9.0)
    s.restore(m)
    assert m.bias.item() == 2.0  # restored the best checkpoint


def test_stopper_min_with_delta_matches_loss_semantics():
    """mode='min', min_delta=1e-4 (val loss): only a strict improvement beyond the margin resets patience."""
    s = EarlyStopper(patience=2, mode="min", min_delta=1e-4)
    assert s.update(1.0, _lin(0.0)) is False                # first -> best
    assert s.update(0.9999, _lin(0.0)) is False             # within min_delta -> NOT an improvement, bad=1
    assert s.bad == 1
    assert s.update(0.5, _lin(0.0)) is False                # real improvement -> bad resets
    assert s.bad == 0
    assert s.best_metric == 0.5


def test_stopper_patience_triggers_stop():
    s = EarlyStopper(patience=2, mode="max")
    s.update(1.0, _lin(0.0))                                # best
    assert s.update(0.5, _lin(0.0)) is False                # bad=1
    assert s.update(0.5, _lin(0.0)) is True                 # bad=2 == patience -> stop


def test_stopper_no_update_restore_is_noop():
    s = EarlyStopper(patience=1, mode="max")
    m = _lin(7.0)
    s.restore(m)                                            # never updated -> keep current weights
    assert m.bias.item() == 7.0
