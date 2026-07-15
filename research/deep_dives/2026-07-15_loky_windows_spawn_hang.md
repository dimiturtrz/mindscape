# loky workers "compute but never return / respawn forever" on Windows — diagnosis

*2026-07-15 · web research · scope: native Win11, sklearn/joblib loky backend, GIL-bound CSP/FBCSP*

## Verdict up front

The "dead end on Windows" conclusion is **premature**. The reported symptom —
loky workers *visibly compute* but *never return* and *appear to respawn
indefinitely*, 0 folds in 200-400 s — is the textbook signature of a **missing
`if __name__ == "__main__":` guard** under Windows `spawn`, not an intrinsic
loky limitation. There is a second, independent, well-documented failure mode
(MKL/OpenMP oversubscription deadlock) that is worth pre-empting at the same
time. Both have standard, cheap fixes. Try them before reverting to `threading`.

---

## Q1 — Is this a known, fixable issue?

Yes. Two distinct, documented root causes both present as "workers run but the
call never returns" on Windows. They are not mutually exclusive; fix both.

### Root cause A (most likely here): missing `__main__` guard → infinite respawn

On Windows there is no `fork`; loky/multiprocessing use **`spawn`**, which
**re-imports the main module** in every worker to reconstruct state. If the
top-level module executes the `Parallel(...)` / `cross_val_score(n_jobs=…)` call
at import time (i.e. it is *not* inside `if __name__ == "__main__":` or a
function called only from such a block), then **each spawned worker re-runs that
same parallel call on import**, spawning its own workers, which spawn more… →
unbounded process fan-out. Each new process *does* start computing (you see
MNE/scipy/MKL banners) but the top-level future never completes, so **0 folds
finish**. This exactly matches "compute but never return / respawn
indefinitely."

Primary sources:
- scikit-learn #10861 "cross_val_score with n_jobs != 1 gets stuck on windows"
  — a core maintainer states you must *"protect the part where you do
  multiprocessing in a `if __name__ == '__main__'` block"*; multiple users
  confirm the guard fixes command-line runs.
  https://github.com/scikit-learn/scikit-learn/issues/10861
- scikit-learn #2433 — same class of Windows hang with `n_jobs>1`, same guard
  remedy. https://github.com/scikit-learn/scikit-learn/issues/2433
- loky spawn implementation re-runs `__main__` as `__mp_main__` and aliases it;
  the guard is what prevents the re-executed module from re-launching the pool.
  loky #236 / joblib #1002 (frozen-exe variants of the same spawn re-import).
  https://github.com/joblib/loky/issues/236 ,
  https://github.com/joblib/joblib/issues/1002

**Important interactive-interpreter corollary:** the same failure happens when
the parallel call is driven from an environment where `__main__` is not a real
importable script — Jupyter/IPython/Spyder consoles, `exec`'d strings, some test
harnesses, or a REPL. #10861 users report it works from the command line but
hangs in Spyder's IPython console. If mindscape's harness invokes the CV from
anything other than a proper `python -m pkg.module` / `python script.py` entry
guarded by `__main__`, that is the prime suspect. Fix = a real module entry
point with the CV call reached only under the guard (loky's cloudpickle handles
`__main__`-defined *functions*, but the *guard* is still required to stop the
respawn).

### Root cause B (independent, also documented): MKL/OpenMP oversubscription deadlock

Nested parallelism — loky spawns N worker *processes*, each of which lets
MKL/OpenMP spawn N *threads* → up to N×N threads contending for N cores. This is
documented to cause not just slowdown but hard **deadlocks**, with workers stuck
in the OpenMP barrier (`gomp_team_barrier_wait_end`).

Primary sources:
- loky #248 "Deadlock in loky worker with OpenMP threads (GOMP)" — py-spy shows
  the worker wedged in the GOMP team barrier; note that *"loky workers just set
  `OMP_NUM_THREADS` and do not mess with the runtime itself."*
  https://github.com/joblib/loky/issues/248
- loky #224 "Deadlock or over-subscription when running the sklearn test suite
  in parallel" — *"Running with `OMP_NUM_THREADS=1` makes the problem go away."*
  https://github.com/joblib/loky/issues/224
- joblib #834 — cases where joblib's automatic BLAS-thread limiting fails to
  take effect. https://github.com/joblib/joblib/issues/834

---

## Q2 / Q3 — Concrete fixes, confirmed vs speculative

### Fix 1 (do first): the `__main__` guard / real entry point — CONFIRMED
Ensure the CV is launched only from:
```python
def main() -> None:
    ...
    cross_val_score(est, X, y, cv=cv, n_jobs=-1)

if __name__ == "__main__":
    main()
```
Nothing that spawns a pool may run at module import time. If driven from a
notebook/REPL, move it into an importable module and call that. This alone
resolves the "respawn forever, 0 folds" pattern in #10861/#2433. (Add
`multiprocessing.freeze_support()` at the top of the guard only if you ever
freeze to an .exe — not needed for a plain interpreter.)

### Fix 2 (do together with Fix 1): cap inner threads — CONFIRMED for the deadlock mode
The scikit-learn parallelism doc (authoritative) explains loky *already* sets
`max_threads = n_cpus // n_jobs` in children from joblib ≥ 0.14, **but** any
manually-exported `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/
`BLIS_NUM_THREADS` *overrides* that heuristic (total becomes
`n_jobs * <LIB>_NUM_THREADS`).
https://scikit-learn.org/stable/computing/parallelism.html

Preferred, scoped mechanism (does not clobber the main process, only children):
```python
from joblib import parallel_config
with parallel_config(backend="loky", inner_max_num_threads=1):
    cross_val_score(est, X, y, cv=cv, n_jobs=-1)
```
`inner_max_num_threads` is documented to set the OpenBLAS/MKL/OpenMP threadpool
cap in the child processes and is loky-only. (Older API:
`joblib.parallel_backend("loky", inner_max_num_threads=1)`.)
https://joblib.readthedocs.io/en/stable/generated/joblib.parallel_config.html

Env-var alternative (blunter — also throttles the main process): set
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1` before import.
Confirmed to clear the deadlock in loky #224/#248. Fine for CSP/FBCSP where the
parallelism you want is across folds (processes), not within-fold BLAS threads.

### Fix 3 (diagnostic knob): `LOKY_MAX_CPU_COUNT` — partial
Caps worker count. Useful to bound fan-out and to work around wrong core
detection, but it does **not** cure the `__main__`-respawn bug — a respawn loop
still overruns any cap eventually. Treat as a safety belt, not the fix.

### Fix 4: `prefer="processes"` vs `"threads"` — not the fix
`prefer="threads"` = the GIL-capped `threading` backend you already fall back to;
it sidesteps the bug by not spawning, at the cost you're trying to avoid. Not a
resolution of the loky problem.

---

## Q4 — MNE / pyriemann + loky on Windows specifically
No single canonical "MNE+pyriemann deadlocks on Windows" issue surfaced, but:
- MNE and pyriemann both themselves call `joblib.Parallel` internally (MNE's
  `n_jobs`, pyriemann estimators). **Nested joblib** (your outer `cross_val_score
  n_jobs=-1` around an estimator that itself parallelizes) multiplies the
  oversubscription in Fix 2 and is a known deadlock amplifier (loky #224 is
  literally the sklearn suite's nested-parallel deadlock). If any inner
  estimator uses `n_jobs>1`, set inner `n_jobs=1` and parallelize only at the CV
  level.
- The heavy per-worker spawn cost you noted (Windows re-imports mne/scipy/mkl per
  worker) is real and expected under `spawn`; it makes short folds look like
  hangs. It inflates wall-time but does not by itself cause a *never-returns*
  loop — that's Fix 1's territory.

---

## Recommended order to try (cheapest signal first)
1. **Wrap the CV entry in `if __name__ == "__main__": main()`** (or move it out of
   the notebook/REPL into a `python -m` module). Single most likely fix.
2. Add `with parallel_config(backend="loky", inner_max_num_threads=1):` around
   the call; set inner-estimator `n_jobs=1`.
3. If still stuck, export `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`
   and retry; use py-spy on a worker to confirm whether it's wedged in
   `gomp_team_barrier_wait_end` (→ oversubscription) vs merely re-importing
   (→ respawn/guard).

## Sources
- https://github.com/scikit-learn/scikit-learn/issues/10861 (Windows n_jobs hang, __main__ guard)
- https://github.com/scikit-learn/scikit-learn/issues/2433 (same, cross_val_score n_jobs>1)
- https://github.com/joblib/loky/issues/248 (GOMP barrier deadlock)
- https://github.com/joblib/loky/issues/224 (oversubscription deadlock, OMP_NUM_THREADS=1 fix)
- https://github.com/joblib/joblib/issues/834 (BLAS-limit mechanism failing)
- https://github.com/joblib/loky/issues/236 , https://github.com/joblib/joblib/issues/1002 (spawn re-import / frozen-exe)
- https://scikit-learn.org/stable/computing/parallelism.html (authoritative: oversubscription, env-var precedence)
- https://joblib.readthedocs.io/en/stable/generated/joblib.parallel_config.html (inner_max_num_threads)
