# Porous Media LBM — Symmetry Discovery Plan

Discover the hidden scaling symmetry in 3D porous media flow data generated
by a Lattice Boltzmann Method (LBM) simulator using the PyDimension Stage1
pipeline.

---

## Problem Statement

**We are fitting:**

```
f  =  g(φ, Re_p)
```

where:

| Symbol | Name | Definition | Dimensionless? |
|---|---|---|---|
| `f` | Friction factor | `(dP_L · d) / (rho · v²)` | **Yes — output** |
| `φ` | Porosity | `n_fluid_nodes / n_total_nodes` | Yes — input |
| `Re_p` | Particle Reynolds | `rho · v · d / mu` | Yes — input |

The ground truth (Ergun equation) is:

```
f  =  [ 150·(1−φ)/Re_p  +  1.75 ] · (1−φ) / φ³
```

The pipeline **does not know this formula** — it recovers the structure
(dimensionless groups, latent dimension, symmetry type) entirely from LBM data.

---

## Inputs and Output — Crystal Clear

### Raw physical variables recorded per LBM run (6 total)

| Column | Symbol | Units | Role | How extracted from LBM |
|---|---|---|---|---|
| `dP_L` | pressure gradient | Pa/m | input | body force × mean density |
| `v` | superficial velocity | m/s | input | `u_x.sum() / N_total` (solid nodes = 0) |
| `mu` | dynamic viscosity | Pa·s | input | `rho · (tau − 0.5) / 3` (lattice units) |
| `rho` | fluid density | kg/m³ | input | `mean(Σ_i f_i)` over fluid nodes |
| `d` | particle diameter | m | input | `2 × sphere_radius` (lattice units) |
| `phi` | porosity | — | input | `n_fluid / n_total` |

### Computed output (dimensionless, 1 column)

| Column | Symbol | Definition | Units |
|---|---|---|---|
| `f` | friction factor | `(dP_L · d) / (rho · v²)` | dimensionless |

> `f` is computed from the 6 raw LBM columns and saved as the target column
> in `dataset_lbm_porous.csv`. The neural network predicts `f` from the
> 5 dimensional inputs `{dP_L, v, mu, rho, d}` plus `phi`.

### Derived reference columns (saved for validation, NOT used as NN inputs)

| Column | Definition | Purpose |
|---|---|---|
| `Re_p` | `rho · v · d / mu` | Ergun collapse plot |
| `f_ergun` | Ergun prediction for `f` | Validate LBM data quality |

---

## Motivation

The Ergun equation shows that `f` depends only on the two dimensionless
combinations `Re_p` and `φ` — not on the 5 individual dimensional inputs
separately. This is evidence of a **scaling symmetry**: any simultaneous
rescaling of `(dP_L, v, mu, rho, d)` that preserves `Re_p` must also preserve
`f`. The Stage1 pipeline recovers this invariance directly from data.

This case is the *cleanest possible test* for the pipeline because:
- Every variable is varied **continuously and independently** (no material
  confounding as in the LPBF 5-alloy case)
- LBM data is **noise-free** — no measurement uncertainty
- The underlying physics is exactly power-law in the pure Darcy regime,
  transitioning to a constant in the Forchheimer regime

---

## Physics Background

### Dimension Matrix

The 5 dimensional inputs (φ is dimensionless, excluded from matrix):

```
         dP_L   v    mu   rho    d
Mass  [   1     0    1     1     0  ]
Len   [  -2     1   -1    -3     1  ]
Time  [  -2    -1   -1     0     0  ]
```

Rank = 3, so by the **Buckingham Pi theorem** (5 dimensional variables − 3
dimensions = **2 independent dimensionless groups**), plus φ already
dimensionless = **3 Pi groups total**.

### Pi Groups

```
π1 = phi                               (porosity — dimensionless on its own)
π2 = rho · v · d / mu                  = Re_p   (particle Reynolds number)
π3 = (dP_L · d) / (rho · v²)           = f      (friction factor — the OUTPUT)
```

The Ergun equation says: `f = g(Re_p, φ)` — so `f` is fully determined by
the other two Pi groups. This means:
- Latent dimension `k* = 2` (Re_p and φ are the two coordinates needed)
- The pipeline should confirm this without being told

---

## Dataset Generation (LBM)

### Geometry

| Parameter | Value |
|---|---|
| Domain size | 64 × 64 × 64 lattice units |
| Boundary conditions | Periodic in all 3 directions |
| Flow drive | Body force in x-direction |
| Geometry type | Random sphere packing (non-overlapping) |

**Why 64³?**
- Sphere radius r = 5 LU → 64/(2×5) ≈ 6 sphere diameters per side — satisfies
  the Representative Elementary Volume (REV) criterion (≥ 5 diameters needed)
- Memory: 64³ × 19 (D3Q19) × 8 bytes ≈ 190 MB per run — manageable
- Run time: ~20–30 s per simulation in optimised NumPy

### Parameter Sweep

| Parameter | Values | Count |
|---|---|---|
| Sphere radius `r` (LU) | 4, 5, 6, 8 | 4 |
| Porosity `phi` | 0.35, 0.40, 0.45, 0.50, 0.55 | 5 |
| Relaxation time `tau` | 0.6, 0.7, 0.9, 1.2, 1.6 | 5 |
| Body force `F` | 1e-6, 5e-6, 1e-5, 5e-5, 1e-4 | 5 |

**Total runs:** 4 × 5 × 5 × 5 = **500 simulations**
**After filtering** Re_p > 300 (turbulent, outside Ergun validity):
expected **~300 usable points**.

### Computing `f` from Each LBM Run

```python
# All quantities in lattice units
nu      = (tau - 0.5) / 3.0                  # kinematic viscosity
rho     = dist_fn.sum(axis=-1).mean()         # mean density
u_x     = (dist_fn * e_x).sum() / (rho * N)  # superficial velocity
mu      = rho * nu                            # dynamic viscosity
dP_L    = F_body * rho                        # pressure gradient (body force)
d       = 2 * sphere_radius                   # particle diameter
phi     = n_fluid / N_total                   # porosity

f       = (dP_L * d) / (rho * u_x**2)        # friction factor — OUTPUT
Re_p    = rho * u_x * d / mu                  # particle Reynolds — reference
```

### Dataset CSV Schema

```
dP_L, v, mu, rho, d, phi, f, Re_p, f_ergun
```

- Columns 1–6: raw physical inputs (fed to Stage1 pipeline)
- Column 7: `f` — **dimensionless output** (target for NN)
- Columns 8–9: reference only (validation, not NN input)

### Quality Checks

- Discard runs with Re_p > 300 (beyond Ergun validity)
- Discard runs not at steady state: `|Δv/v| < 1e-5`
- Pre-pipeline validation: plot `f · φ³/(1−φ)` vs `Re_p/(1−φ)` — all points
  should collapse onto `150/x + 1.75` with R² > 0.98 before running Stage1

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi Reduction)

**Input:** The 3×5 dimension matrix (5 dimensional variables; φ excluded as
already dimensionless).

**What it does:**
1. Computes the null space via `pydimension.data_preprocessing.DataPreprocessor`
2. Recovers the 2 primitive integer Pi-group basis vectors from dimensional vars
3. φ is appended as the third Pi feature
4. Appends `log10` of each Pi group as reduced candidates for Step 2

**Expected output — 3 Pi groups:**
```
π1 = phi                               (porosity)
π2 = rho · v · d / mu                  (Re_p)
π3 = dP_L · d · rho⁻¹ · v⁻²           (friction factor f — the output)
```

**Verification:** Project known Ergun exponent vector onto null-space →
cos = +1.0000

---

### Step 1 — Normalisation

**Inputs:** Raw X (n × 6, all six variables including φ) and output y = f (n,).

- **Min-max scaling** of raw X → `X_norm_raw` (for Step 3)
- **Min-max scaling** of Pi features → `X_norm_step2` (for Step 2, pi-only mode)
- **Min-max scaling** of f → `y_norm`

---

### Step 2 — Latent Dimension Discovery

**Goal:** Find the smallest `k*` needed to predict `f`.

**Input (pi-only mode):** 2 Pi features: `[log10(Re_p), φ]` scaled to [0, 1].

> Note: φ is included directly (already dimensionless). `f` is the output —
> not fed as an input.

**Encoder:** Multilayer MLP (hidden dims `[64, 32]`):
```
2  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```

Sweep over `k = 1, 2, 3`.

**Expected results:**

| k | Expected R² (test) | Interpretation |
|---|---|---|
| 1 | ~0.85 | Re_p alone (misses φ dependence) |
| **2** | **~0.99 ← optimal** | Both Re_p and φ needed |
| 3 | ~0.99 + overfitting | No improvement |

**`k* = 2`** — confirms `f = g(Re_p, φ)`.

---

### Step 3 — Symmetry Type Identification

**Goal:** Confirm the invariance is a scaling (power-law) symmetry.

**Input:** `X_norm_raw` — all 6 raw physical variables (always used here, not
Pi groups, so the log/X/X² transforms act on multiplicatively-meaningful quantities).

| Encoder | Transform | Tests |
|---|---|---|
| Scaling | `z = W · log\|X\|` | Power-law / dimensional symmetry |
| Translational | `z = W · X` | Additive / affine symmetry |
| Rotational | `z = W · X²` | Quadratic / Euclidean symmetry |

**Expected:**
```
scaling        : ~0.005   ← winner
translational  : ~0.05
rotational     : ~0.08
Loss gap: ≥ 10×
```

---

### Step 4 — Generator Extraction

**Input:** Winning encoder weight matrix `W` (shape `2 × 6`).

With `k* = 2`: `6 − 2 = 4` null-space generators — all fully determined
(every variable varies continuously in the sweep).

---

### Step 5 — Physical Interpretation

**Predicted generators** (directions in log-space that leave `f` invariant):

| Generator | Trade-off | Physical meaning |
|---|---|---|
| G1 | ρ ↑, μ ↑ (same ratio) | Fluid swap at fixed Re_p: denser + more viscous fluid, same flow resistance |
| G2 | d ↑, v ↓ (proportional) | Length-scale rescaling at fixed Re_p |
| G3 | dP_L ↑, ρ ↑, v² ↑ | Inertial invariance: scale up pressure and inertia together |
| G4 | dP_L ↑, μ ↑, v ↑ | Darcy invariance: `v/dP_L = const` at fixed geometry |

**Key validations:**
- **Kozeny–Carman exponents** `(0, 0, 0, 0, 2, 3)` on `(dP_L, v, mu, rho, d, phi)`
  projected onto null space → cos ≈ +1
- **Ergun exponent vector** projected onto Pi null-space → cos = +1.0000

---

## Output Files

All outputs go to `output_porous_media_lbm_symmetry/`.

| File | Contents |
|---|---|
| `dataset_lbm_porous.csv` | ~300 rows: 6 raw inputs + `f` (output) + `Re_p`, `f_ergun` (reference) |
| `lbm_ergun_collapse.png` | Validation: `f·φ³/(1−φ)` vs `Re_p/(1−φ)` — all points on `150/x + 1.75` |
| `lbm_pi_candidates.png` | Pi-basis heatmap + scatter of `f` vs each `log₁₀(Πₖ)` |
| `lbm_symmetry_discovery.png` | 2-panel: symmetry-type bar chart + latent-dim R² curve |
| `_da_repo/dimension_matrix.csv` | Explicit dimension matrix fed to DataPreprocessor |
| `_da_repo/basis_vectors.csv` | Integer Pi-group exponent vectors |

---

## Expected Summary of Results

| Aspect | Expected Result |
|---|---|
| **Output variable** | `f = (dP_L · d) / (rho · v²)` — dimensionless |
| **Correlation fitted** | `f = g(Re_p, φ)` |
| Pi groups discovered | 3: φ, Re_p, f |
| Known Ergun Pi in null-space | cos = +1.0000 ✓ |
| Latent dimension k* | **2** — Re_p and φ both needed |
| Test R² | **≥ 0.99** (noise-free LBM data) |
| Symmetry type | **Scaling** — ≥ 10× loss gap |
| Generators | 4 directions (6 variables − 2 latent dims), all constrained |

---

## File Organisation

```
PyDimension/
├── pydimension/
└── projects/20260912_Stage1_Prokash/
    └── Examples/porous_media_lbm_symmetry/
        ├── PLAN.md                           ← this file
        ├── generate_dataset.py               ← LBM sweep script (TODO)
        ├── discover_symmetry.py              ← Stage1 pipeline script (TODO)
        └── output_porous_media_lbm_symmetry/
```

---

## Next Steps

1. **Share LBM code interface** — function signature for one simulation run
   (inputs: geometry, τ, body force; outputs: steady-state velocity field)
2. **Write `generate_dataset.py`** — runs the parameter sweep, computes `f`
   and `Re_p` per run, saves `dataset_lbm_porous.csv`
3. **Write `discover_symmetry.py`** — Stage1 pipeline for this 6-variable
   problem (follows `lpbf_porosity_symmetry/discover_symmetry.py` structure)
4. **Validate Ergun collapse** — verify R² > 0.98 on Ergun plot before Stage1
5. **Run full symmetry discovery** — confirm `k* = 2`, scaling wins, 4 generators

---

## Dependencies

`torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.
Install with `pip install -r requirements.txt` from the repo root.
