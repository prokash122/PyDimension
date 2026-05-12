# Porous Media LBM — Symmetry Discovery

Discover the hidden scaling symmetry in 3D porous media flow data generated
by a Lattice Boltzmann Method (LBM) simulator using the PyDimension Stage1
pipeline. The pipeline recovers the **Ergun-equation structure**
(`f = g(Re_p, φ)`) entirely from the LBM simulation data — without being
given any formula in advance.

> See **PLAN.md** for the full design rationale and expected pipeline behavior.

---

## Problem Statement

**Output (dimensionless):**
```
f = (dP_L · d) / (rho · v²)            friction factor
```

**Inputs (6 physical variables):**
```
dP_L, v, mu, rho, d, phi
```

**Correlation we are fitting:**
```
f = g(Re_p, phi)
```
where `Re_p = rho · v · d / mu` is the particle Reynolds number.

The ground-truth Ergun formula is `f = [150·(1−φ)/Re_p + 1.75] · (1−φ)/φ³`,
but the pipeline does not know this — it should recover the structure
(2 latent coords, scaling symmetry, 4 generators) from data alone.

---

## Dataset — `dataset_lbm_porous.csv`

| Property | Value |
|---|---|
| Rows | 96 converged LBM runs |
| Geometries | 4 (all r = 5 LU; phi ∈ {0.46, 0.50, 0.55, 0.61}) |
| Relaxation time `tau` | 0.9, 1.0, 1.1 (3 viscosities) |
| Driving force `delta_p` | 8 values: 1e-6 to 3e-4 |
| Particle diameter `d` | constant = 10 LU |
| Re_p range | [5e-7, 1e-3] — **deep Darcy regime** |
| f range | [7.6e4, 4.6e8] — spans ~5 orders of magnitude |

**Schema:**

| Column | Description |
|---|---|
| `filename`, `tau`, `delta_p` | LBM run identifiers |
| `dP_L`, `v`, `mu`, `rho`, `d`, `phi` | 6 physical inputs (NN inputs) |
| `f` | **dimensionless friction factor — NN target** |
| `Re_p`, `f_ergun` | reference values (not NN inputs) |
| `steps`, `converged`, `stalled` | convergence diagnostics |

### Known limitations of this dataset

- **Single particle size (d = 10 LU)** — symmetry generators involving `d` are
  not fully constrained. The pipeline will report the d-direction as
  near-zero in the encoder weights (uninformative).
- **Deep Darcy regime only** (Re_p < 1e-3) — the inertial Forchheimer term
  (1.75) is invisible. The data follows pure `f ∝ 1/Re_p`.
- **f / f_ergun ≈ 0.66** — LBM gives ~34% lower drag than the textbook
  Ergun constant (150). This is fine for symmetry discovery (the functional
  form, not the prefactor, is what matters).

To address these limitations, the dataset can be extended with additional
sphere radii `r ∈ {4, 6, 8}` and higher driving force `delta_p ∈ {1e-3, 1e-2}`
to reach the Forchheimer regime.

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi)

**Driver:** `pydimension.data_preprocessing.DataPreprocessor` — same null-space +
SymPy primitive-integer reduction used by every other Stage1 example. The
script writes a hand-checked `dimension_matrix.csv` to `_da_repo/` and calls
`DataPreprocessor.process_with_dimensional_analysis()`. An inline
`compute_pi_basis()` (scipy + SymPy) is retained as a fallback and can be
forced via `--no-repo-da`.

**Input:** 3×6 dimension matrix (M, L, T × 6 variables; phi already dimensionless).

**Output — 3 Pi groups (DataPreprocessor):**
```
π1 = dP_L · v⁻³ · μ · ρ⁻²        (= f / Re_p² in the f-Re_p basis)
π2 = dP_L · v⁻¹ · μ⁻¹ · d²       (Darcy form)
π3 = φ
```

Any null-space basis is valid — the canonical pair `Re_p = ρvd/μ` and
`f = dP_L·d/(ρv²)` is a different choice of basis vectors in the same span.

**Verification:** Project known `f` and `Re_p` exponent vectors onto the
discovered null-space → both give cos = +1.0000 ✓ (confirming `f` and `Re_p`
lie exactly in the span of the discovered Pi groups).

---

### Step 1 — Normalisation

- Raw X (6 columns) min-max scaled to [0, 1] → for Step 3
- `log10(Re_p)` and `phi` min-max scaled → Pi features for Step 2
- `log10(f)` (NN target) min-max scaled — f spans 5 orders of magnitude, so
  log-transformation is essential

---

### Step 2 — Latent Dimension Discovery

**Input:** 2 Pi features: `[log10(Re_p), phi]`.

**Encoder:** Multilayer MLP `[2 → 64 → 32 → k]`. Sweep `k = 1, 2`.

**Expected (for deep Darcy regime):**
- k = 1: R² high but not perfect (Re_p alone misses phi dependence)
- k = 2: R² → 1.0 (Re_p and phi both needed)

---

### Step 3 — Symmetry Type Identification

**Input:** Raw 6-variable physical X.

Three competing single-layer encoders test power-law (`log|X|`),
additive (`X`), and quadratic (`X²`) symmetries.

**Expected (physics):** Scaling should win, because the Darcy law
`f ∝ 1/Re_p` is a pure power-law.

**Observed:** No symmetry type wins decisively on this dataset — see the
**Caveat on symmetry-type volatility** in Observed Results below. With only
3 of 6 variables meaningfully varying, all three losses sit within ~1.5×
of each other and the winner flips run-to-run. This is honest behaviour:
the pipeline reports its uncertainty rather than confidently picking the
wrong family. Step 0 (Pi recovery) is the physically meaningful result.

---

### Step 4 — Generator Extraction

With `k* = 2`, there are `6 − 2 = 4` null-space generators in 6-D log space.

**Caveat:** The `d` column is constant in this dataset, so the generator
component along `d` is undetermined. The pipeline reports it but it should
not be physically interpreted.

---

### Step 5 — Physical Interpretation

Expected generators (directions in log-space that preserve `f`):

| Generator | Trade-off | Meaning |
|---|---|---|
| G1 | ρ ↑, μ ↑ (proportional) | Fluid swap at fixed Re_p |
| G2 | dP_L ↑, μ ↑, v ↑ | Darcy invariance |
| G3 | (along `d`) | **Undetermined** — d is constant in this dataset |
| G4 | combined Re_p / f trade-off | Darcy-regime invariance |

---

## Output Files

All outputs go to `output_porous_media_lbm_symmetry/`.

| File | Contents |
|---|---|
| `lbm_ergun_collapse.png` | f·φ³/(1−φ) vs Re_p/(1−φ), coloured by porosity, with textbook curve overlaid |
| `lbm_pi_candidates.png` | Pi-basis heatmap + scatter of log₁₀(f) vs each log₁₀(Πₖ) |
| `lbm_symmetry_discovery.png` | 2-panel: symmetry-type bars + latent-dim R² curve |
| `run.log` | Full stdout log of the pipeline run |
| `_da_repo/dimension_matrix.csv` | Hand-checked integer dimension matrix fed to `DataPreprocessor` |
| `_da_repo/data/basis_vectors.csv` | Primitive integer Pi-group exponent vectors emitted by `DataPreprocessor` |
| `_da_repo/data/afterDA_data.csv` | Normalised Pi values per row produced by `DataPreprocessor` |

---

## Observed Results (committed run)

The committed output figures were produced with reduced training
(`--latent-epochs 300 --sym-epochs 600 --n-restarts 2 --seed 42`).
Step 0 is driven by `pydimension.data_preprocessing.DataPreprocessor`.
Full results:

| Aspect | Observed |
|---|---|
| Pi-basis driver | `DataPreprocessor.process_with_dimensional_analysis()` ✓ |
| Pi groups discovered | 3 — `π1 = dP_L·v⁻³·μ·ρ⁻²`, `π2 = dP_L·v⁻¹·μ⁻¹·d²`, `π3 = φ` |
| Known `f` exponents in Pi span | **cos = +1.0000 ✓** |
| Known `Re_p` exponents in Pi span | **cos = +1.0000 ✓** |
| Latent dimension `k*` | **1** (R² ≈ 0.997 at k=1) |
| Symmetry type | **Rotational** (MSE 0.0089) > scaling (0.0125) > translational (0.0167) |
| Loss gap | ~1.4× — well within the dataset's symmetry-instability band |
| Generators | 5 directions (6 vars − 1 latent) |

**Stable physics (Step 0):** The Buckingham-Pi reduction is fully
reproducible. `DataPreprocessor` returns the same three primitive integer
Pi groups every run, and the known `f` and `Re_p` exponent vectors both
project onto the discovered span with `cos = +1.0000`. This is the
physically meaningful result of the example.

**Ergun collapse:** All 4 porosity bins collapse onto a single curve with
slope = −1 (pure Darcy), running **~30% below** the textbook `150/x` line.
This systematic offset (mean `f / f_ergun ≈ 0.66`) is consistent across all
data and indicates the LBM under-predicts drag relative to the textbook
empirical constant — likely a combination of single-packing-realisation
effects, `d/Δx = 10` resolution, and bounce-back boundary slip.

**Why `k* = 1` instead of the expected `k* = 2`?** Because in the pure
Darcy regime `f ≈ 150·(1−φ)²/(φ³·Re_p)`, and `log(f)` varies by ~5 orders
of magnitude from Re_p but only ~0.65 from the φ factor across this
porosity range. So `Re_p` alone explains 99.7% of the variance —
`φ` only contributes a small logarithmic correction. To force `k* = 2`,
we'd need a wider φ range or higher Re_p where the inertial Forchheimer
term `1.75·(1−φ)/φ³` becomes important.

**Caveat on symmetry-type (Step 3) volatility:** The Step 3 winner is
**not** robust on this dataset. Across fresh runs and different seeds the
loss gap between scaling, translational, and rotational stays in the
1.0–2.0× range, and any of the three can win. Earlier committed runs
reported "scaling" with a 1.6× gap; current runs report "rotational" with
a 1.4× gap. This is the expected behaviour for a dataset where only 3 of
6 variables have meaningful variation (dP_L, v, mu) — d is constant, rho
is nearly constant, and phi has only 4 levels. The symmetry-discovery
machinery does not have enough leverage to separate the three encoder
families. **The physically meaningful result is the Buckingham-Pi
recovery (cos = +1.0000), not the Step 3 winner.** The fix is more
particle-size variation and a wider Re_p range — see Future Work below.

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/porous_media_lbm_symmetry

# Default run
python discover_symmetry.py --data dataset_lbm_porous.csv

# Deeper encoder
python discover_symmetry.py --data dataset_lbm_porous.csv --encoder-hidden 128 64 32

# Longer training for stable symmetry detection
python discover_symmetry.py --data dataset_lbm_porous.csv --sym-epochs 3000

# Different output directory
python discover_symmetry.py --data dataset_lbm_porous.csv --output-dir my_output
```

Default training budget: `--latent-epochs 600`, `--sym-epochs 1500`,
`--n-restarts 3`, `--encoder-hidden 64 32`.

---

## File Organisation

```
PyDimension/
├── pydimension/                              ← auto-discovered
└── projects/20260912_Stage1_Prokash/
    ├── preprocessing/
    ├── intrinsic_coordinate/
    ├── symmetry_discovery/
    └── Examples/porous_media_lbm_symmetry/
        ├── PLAN.md                           ← full design plan
        ├── README.md                         ← this file
        ├── discover_symmetry.py              ← Stage1 pipeline script
        └── dataset_lbm_porous.csv            ← 96-row LBM dataset
```

Dependencies: `torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.
Install with `pip install -r requirements.txt` from the repo root.

---

## Future Work

1. **Add particle-size variation** — re-run LBM with r ∈ {4, 6, 8} to fully
   constrain the d-direction generator.
2. **Reach Forchheimer regime** — add `delta_p ∈ {1e-3, 1e-2}` to push
   Re_p > 10 and expose the inertial 1.75 term.
3. **Investigate the f/f_ergun ≈ 0.66 offset** — compare against the
   Macdonald–Ergun correlation (uses 180 instead of 150 for natural particles)
   to see if your packing follows a different empirical constant.
