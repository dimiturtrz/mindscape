# M2 · Linear algebra → spatial filtering & CSP

Working fluency, tied to where it bites. Reference: **3Blue1Brown — *Essence of Linear Algebra*** (esp.
ch. 14, *Eigenvectors and eigenvalues* — the geometric intuition); **StatQuest** (*Covariance*, *PCA*).

## The core ideas (covariance → CSP, derived)
- **Covariance matrix.** For a trial `X ∈ ℝ^{C×T}` (zero-mean), `Σ = (1/T) X Xᵀ`. Each time-sample is one
  observation of the C-channel vector → `Σ` summarizes the cloud of T points in C-D space. **Diagonal =
  per-channel variance = (post-bandpass) band power**; off-diagonal = co-movement (mostly volume conduction,
  zero-lag → high covariance). Raw covariance, not correlation (CSP wants the power scale). Units V² = power.
- **Filter power = a quadratic form.** A spatial filter `w ∈ ℝ^C` gives a virtual channel `wᵀX`; its power is
  **`var(wᵀX) = wᵀ Σ w`** (derived: `(wᵀxₜ)² = wᵀ(xₜxₜᵀ)w`, average → `wᵀΣw`). So `Σ` is a power-calculator
  for any filter, no re-filtering.
- **Objective = Rayleigh quotient.** Max power-contrast between classes: `J(w) = wᵀC₁w / wᵀC₂w`. Setting
  `∇J=0` gives **`C₁w = λC₂w`** (generalized eigenproblem); `λ = J(w)` = the power ratio. So the optimal
  filters ARE the generalized eigenvectors, ranked by exactly the contrast we wanted.
- **22 eigenvectors.** A symmetric C×C matrix has C orthogonal eigenvectors (= the C principal axes of the
  C-D data ellipsoid). `∇J=0` catches ALL critical points: one max (λ≈1, left detector), one min (λ≈0,
  right detector), the rest **saddle points** (λ≈0.5, equal power, non-discriminative). Keep extremes →
  `n_components=6` (3 top + 3 bottom); drop the saddles.
- **λ meaning.** Composite form `λ = wᵀC₁w / wᵀ(C₁+C₂)w` = fraction of the filter's power that's class-1.
- **Spectral theorem (the foundation under it).** Real symmetric ⟹ real eigenvalues + a FULL orthonormal
  eigenbasis + orthogonally diagonalizable (`A=QΛQᵀ`). NOT "full rank" (separate), and "N eigenvalues" is
  true of all matrices — the symmetric gift is *real + orthogonally complete*. This is why CSP's 22 real
  orthogonal eigenvectors exist at all; it only works because covariance is symmetric.
- **Whitening = generalized → standard.** `P = D^{-1/2}Uᵀ` from `C₁+C₂ = UDUᵀ` (i.e. `P ≈ C_c^{-1/2}`, the
  matrix "divide by the std-dev"). Then `P(C₁+C₂)Pᵀ = I` (by construction); substituting `w=Pᵀy` turns
  `C₁w=λC_c w` into the standard `(PC₁Pᵀ)y = λy`. Bonus via linearity: `S₁+S₂ = P(C₁+C₂)Pᵀ = I` ⟹
  `S₂ = I−S₁` — diagonalize one class, the other (and the contrast) falls out free. "Sphere" = the data
  whose covariance is `I` (unit variance every direction, no correlation); nothing mystical.
- **Pipeline.** `C₁,C₂ → solve C₁w=λ(C₁+C₂)w → keep 6 extreme λ → fᵢ = log(wᵢᵀΣwᵢ) → LDA.` 2-class at
  heart; 4-class 2a handled by one-vs-rest / joint diagonalization (MNE's multiclass CSP).

---

## Quiz log

### 2026-06-30 — quiz M2 (then an extended teach-through)
**Initial cold score ~2/6** — knew the mechanics (dot product = linear combination), but the *connections*
were missing. Then a long derivation session brought it to a genuinely solid, bottom-to-top understanding.

Cold answers: (1) confused covariance with correlation (diagonal = variance, not 1) · (2) ✓ dot product /
linear combination; didn't know it undoes volume conduction · (3) eigenvectors fuzzy ("encode the matrix
simply"; "big eigenvalue = irregular") · (4) CSP move not understood · (5) ✓ gist (PCA general vs specific)
· (6) variance-ratio = ERD contrast not understood.

**Closed through teaching** (all landed): diagonal = variance = band power; `var(wᵀX)=wᵀΣw` derived;
Rayleigh quotient → `∇=0` → generalized eigenproblem; 22 eigenvectors = 22 ellipsoid axes; min&max&**saddles**
(why λ≈0.5 exists); spectral theorem (symmetry ⟹ real + orthogonal eigenbasis; not full rank); whitening as
`P=C_c^{-1/2}` making `C₁+C₂=I`, `S₂=I−S₁` by linearity. Connected to prior DSP: **CSP-whitening ≈ GCC-PHAT**
(normalize magnitude/variance so the loud common part doesn't drown the structure).

**Resources flagged:** 3B1B *Essence of LA* (ch. 14 eigenvectors) · StatQuest *Covariance* + *PCA*.

**Assessment:** started shaky, ended fluent — derived CSP end to end including *why* symmetry guarantees the
eigenstructure and *why* whitening makes the sum the identity. Real understanding, not memorized.
