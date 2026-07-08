"""Train/val/test splits as criteria over the epoch cloud (data/store.py).

The leakage-control story lives here. A split isn't a named thing — it's the cloud filtered on criteria, and
the criteria self-document what was held out. The *evaluation regime* IS which criteria you filter on:

    within-subject   train+test one subject (random val carve)      -> the ceiling
    cross-subject    hold out whole subjects as test (LOSO)          -> the OOD gap (the headline)
    cross-session    hold out a session as test                      -> the drift

Same `(fit_fn, score_fn)` harness contract across all three; only the criteria change. (The siblings'
make_split holds out whole datasets/vendors; ours holds out subjects/sessions — same idea, EEG axis.)
"""
from __future__ import annotations

import polars as pl
from pydantic import BaseModel
from sklearn.model_selection import GroupKFold


class SplitSpec(BaseModel):
    """The criteria that define one split. `test_subjects`/`test_sessions` = held-out test rows (by subject
    OR session); `val_subjects` = a held-out subject for tuning (not test) — if empty, a random `val_frac` is
    carved from the non-test rows instead; `seed` makes that carve deterministic."""
    model_config = {"arbitrary_types_allowed": True}
    test_subjects: tuple = ()
    test_sessions: tuple = ()
    val_subjects: tuple = ()
    val_frac: float = 0.2
    seed: int = 0


def _val_carve(rest: pl.DataFrame, val_frac: float, seed: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Deterministic random val carve from `rest` (epoch-level)."""
    shuffled = rest.sample(fraction=1.0, shuffle=True, seed=seed)
    n_val = max(1, round(len(shuffled) * val_frac))
    return shuffled[n_val:], shuffled[:n_val]


def make_split(meta: pl.DataFrame, spec: SplitSpec | None = None
               ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(train, val, test) from the criteria in `spec` (see SplitSpec).

    test = rows whose subject ∈ test_subjects OR session ∈ test_sessions.
    val  = rows whose subject ∈ val_subjects (a held-out subject for tuning, not test) — or, if none
           given, a random `val_frac` carved from the non-test rows (in-distribution val).
    train = everything that's neither test nor val.
    """
    spec = spec or SplitSpec()
    test_subjects = [str(x) for x in spec.test_subjects]
    test_sessions = [str(x) for x in spec.test_sessions]
    val_subjects = [str(x) for x in spec.val_subjects]

    test_expr = (pl.col("subject").is_in(test_subjects) | pl.col("session").is_in(test_sessions))
    test = meta.filter(test_expr)
    rest = meta.filter(~test_expr)
    if val_subjects:
        val_expr = pl.col("subject").is_in(val_subjects)
        return rest.filter(~val_expr), rest.filter(val_expr), test
    train, val = _val_carve(rest, spec.val_frac, spec.seed)
    return train, val, test


def leave_one_subject_out(meta: pl.DataFrame):
    """Yield (held_subject, train, test) for each subject — the cross-subject regime. test = the held
    subject; train = ALL other subjects, in full. No fold-level val carve: the harness discards it anyway,
    and the DL decoders carve their own val from `train` internally (early stopping), so carving here just
    threw training data away — worse for the classical baselines, redundant for the nets."""
    for sub in sorted(meta["subject"].unique().to_list()):
        s = str(sub)
        yield sub, meta.filter(pl.col("subject") != s), meta.filter(pl.col("subject") == s)


def grouped_kfold(meta: pl.DataFrame, k: int = 5):
    """Yield (fold_name, train, test) for k-fold cross-SUBJECT CV — subjects partitioned into k test
    groups, each subject tested exactly once. This is the BenchNIRS 'generalised' protocol (sklearn
    GroupKFold); LOSO is the k = n_subjects limit. train = all non-test subjects, in full (no val carve;
    see leave_one_subject_out)."""
    subs = sorted(meta["subject"].unique().to_list())
    gkf = GroupKFold(n_splits=k)
    for i, (_tr, te) in enumerate(gkf.split(list(range(len(subs))), groups=subs)):
        test_subs = [subs[j] for j in te]
        in_test = pl.col("subject").is_in(test_subs)
        yield f"fold{i}", meta.filter(~in_test), meta.filter(in_test)   # full train (no val carve)


def within_subject(meta: pl.DataFrame, subject: str, test_sessions=(), val_frac: float = 0.2,
                   seed: int = 0) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """One subject in isolation. If the dataset has distinct train/eval sessions, pass the eval
    session(s) as `test_sessions` (the standard 2a protocol); else a random `val_frac`/test carve."""
    one = meta.filter(pl.col("subject") == str(subject))
    if test_sessions:
        return make_split(one, SplitSpec(test_sessions=tuple(test_sessions), val_frac=val_frac, seed=seed))
    # no session protocol: carve test then val from the remainder
    shuffled = one.sample(fraction=1.0, shuffle=True, seed=seed)
    n_test = max(1, round(len(shuffled) * 0.2))
    test, rest = shuffled[:n_test], shuffled[n_test:]
    train, val = _val_carve(rest, val_frac, seed)
    return train, val, test


def sessions(meta: pl.DataFrame) -> list[str]:
    return sorted(meta["session"].unique().to_list())
