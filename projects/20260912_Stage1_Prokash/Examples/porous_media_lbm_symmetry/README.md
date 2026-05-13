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
| Rows | 144 converged LBM runs |
| Sphere radii | 2 (r = 3 and r = 5 LU; d = 6 and d = 10) |
| Porosities | 6 (phi ∈ {0.459, 0.502, 0.543, 0.551, 0.603, 0.609}) |
| Relaxation time `tau` | 0.9, 1.0, 1.1 (3 viscosities) |
| Driving force `delta_p` | 8 values: 1e-6 to 3e-4 |
| Re_p range | [3e-7, 1e-3] — **deep Darcy regime** |
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

- **Only two particle sizes (d ∈ {6, 10})** — the d-direction generator
  is now constrained (it was undetermined in the earlier single-radius
  dataset) but a third radius would still help.
- **Deep Darcy regime only** (Re_p < 1e-3) — the inertial Forchheimer term
  (1.75) is invisible. The data follows pure `f ∝ 1/Re_p`.
- **f / f_ergun ≈ 0.66** — LBM gives ~34% lower drag than the textbook
  Ergun constant (150). This is fine for symmetry discovery (the functional
  form, not the prefactor, is what matters).

To address remaining limitations, the dataset can be extended with one
more sphere radius (e.g. `r = 4` or `r = 7`) and higher driving force
`delta_p ∈ {1e-3, 1e-2}` to reach the Forchheimer regime.

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
(`--latent-epochs 300 --sym-epochs 600 --n-restarts 3 --seed 42`) and
locked threads + hash seed for bit-reproducibility:
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
```
Step 0 is driven by `pydimension.data_preprocessing.DataPreprocessor`.

| Aspect | Observed |
|---|---|
| Pi-basis driver | `DataPreprocessor.process_with_dimensional_analysis()` ✓ |
| Pi groups discovered | 3 — `π1 = dP_L·v⁻³·μ·ρ⁻²`, `π2 = dP_L·v⁻¹·μ⁻¹·d²`, `π3 = φ` |
| Known `f` exponents in Pi span | **cos = +1.0000 ✓** |
| Known `Re_p` exponents in Pi span | **cos = +1.0000 ✓** |
| Latent dimension `k*` | **1** (R² ≈ 0.997 at k=1) |
| Symmetry type | **Scaling ✓** (MSE 0.000924) — winner by **5.0× margin** over translational (0.00460) and rotational (0.00614) |
| Generators | 5 directions in 6-D log-space — Darcy invariance, length-scale rescaling, fluid-swap, plus the under-constrained ρ direction |

**Stable physics (Step 0):** The Buckingham-Pi reduction is fully
reproducible. `DataPreprocessor` returns the same three primitive integer
Pi groups every run, and the known `f` and `Re_p` exponent vectors both
project onto the discovered span with `cos = +1.0000`. **This is the
physically meaningful result of the example.**

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

**Scaling wins decisively across all six seeds:**

| Seed | Winner | scaling MSE | trans MSE | rot MSE | gap |
|---|---|---|---|---|---|
| 42  | **scaling** | **0.000924** | 0.004599 | 0.006139 | 5.0× |
| 0   | **scaling** | **0.003283** | 0.009857 | 0.010836 | 3.0× |
| 1   | **scaling** | **0.003573** | 0.007249 | 0.009900 | 2.0× |
| 2   | **scaling** | **0.002314** | 0.005995 | 0.012743 | 2.6× |
| 7   | **scaling** | **0.001862** | 0.005840 | 0.005315 | 2.9× |
| 100 | **scaling** | **0.003560** | 0.009357 | 0.010138 | 2.6× |

This is the physically expected result: Darcy's law `f ∝ 1/Re_p` is a
pure power-law, so `log f` is linear in `log(ρ·v·d/μ)`, which is
exactly the family of relationships the scaling encoder
`z = W · log|X|` is designed to recover.

**Why earlier runs got "rotational":** The Stage1 scaling encoder
applies `log(|X|.clamp(min=0.1))` internally. The clamp threshold of
0.1 is calibrated for **raw multiplicatively-meaningful X** (positive
values of order unity, spanning a few orders of magnitude). If we
min-max-normalise X into `[0, 1]` first — the default Stage1
preprocessing for the other two examples — many values land below
0.1 and get pinned by the clamp, destroying the multiplicative signal
that the scaling encoder needs. The rotational (`X²`) and
translational (`X`) encoders are bijective on `[0, 1]` so they keep
the full input information, and a flexible MLP decoder approximates
the needed `log()` internally — letting them appear to win on raw fit
MSE even though their feature maps don't encode the underlying
invariance.

The fix in this example: **skip the min-max step on X for Step 3**.
The script geometric-mean-centres each column (so the per-column
geometric mean is exactly 1.0) and passes that directly to
`identify_symmetry`. Values now span roughly `[0.05, 25]` per column
— the regime the clamp threshold was designed for — and the scaling
encoder works as intended. Toggle this with `--no-log-normalize` to
reproduce the broken behaviour for comparison (it will revert to
rotational/translational wins).

**Reproducibility:** lock the three BLAS thread vars and
`PYTHONHASHSEED` so the run is bit-reproducible. Without
`PYTHONHASHSEED=0`, Python's string hash randomisation feeds different
torch seeds into `identify_symmetry` each invocation (it computes
`seed + hash(sym_type) % 1000 + restart * 37`).

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/porous_media_lbm_symmetry

# Reproducible run (matches committed run.log)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python discover_symmetry.py --data dataset_lbm_porous.csv \
    --latent-epochs 300 --sym-epochs 600 --n-restarts 3 --seed 42

# Default run (multi-threaded, faster, but Step 3 winner is non-deterministic)
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
        └── dataset_lbm_porous.csv            ← 144-row LBM dataset (r ∈ {3, 5})
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
