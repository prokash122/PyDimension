# Porous Media LBM — Symmetry Discovery Plan

Discover the hidden scaling symmetry in 3D porous media flow data generated
by a Lattice Boltzmann Method (LBM) simulator using the PyDimension Stage1
pipeline.

---

## Motivation

Flow through porous media is governed by the **Ergun equation** — a
power-law relationship between a dimensionless friction factor and a particle
Reynolds number:

```
f · φ³/(1 − φ) = 150·(1 − φ)/Re_p  +  1.75
```

where the two terms correspond to the **viscous (Darcy) regime** at low Re
and the **inertial (Forchheimer) regime** at high Re.

This relationship is evidence of a **scaling symmetry**: any simultaneous
rescaling of (ΔP/L, v, μ, ρ, d, φ) that preserves Re_p and f must also
preserve the flow velocity. The Stage1 pipeline recovers this invariance
directly from LBM simulation data, without being given the Ergun formula.

This case is the *cleanest possible test* for the pipeline because:
- Every variable is varied **continuously and independently** (no material
  confounding as in the LPBF 5-alloy case)
- LBM data is **noise-free** — no measurement uncertainty
- The underlying physics is **exactly power-law**, so the scaling symmetry
  signal should be unambiguous

---

## Physics Background

### Variables

| Variable | Symbol | SI Units | Dimensions |
|---|---|---|---|
| Pressure gradient | `dP_L` | Pa/m | kg·m⁻²·s⁻² |
| Superficial velocity | `v` | m/s | m·s⁻¹ |
| Dynamic viscosity | `mu` | Pa·s | kg·m⁻¹·s⁻¹ |
| Fluid density | `rho` | kg/m³ | kg·m⁻³ |
| Particle/pore diameter | `d` | m | m |
| Porosity | `phi` | — | dimensionless |

### Dimension Matrix

```
         dP_L   v    mu   rho    d   phi
Mass  [   1     0    1     1     0    0  ]
Len   [  -2     1   -1    -3     1    0  ]
Time  [  -2    -1   -1     0     0    0  ]
```

Rank = 3, so by the **Buckingham Pi theorem** (6 variables − 3 dimensions):
**3 independent dimensionless groups**.

### Pi Groups

```
π1 = phi                               (porosity — dimensionless on its own)
π2 = rho · v · d / mu                  = Re_p  (particle Reynolds number)
π3 = (dP_L · d) / (rho · v²)           = f     (friction factor)
```

**Verification:** The known Ergun exponent vector should project onto the
discovered null-space basis with **cos = +1.0000**.

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
| Body force `F` | 1e-6, 5e-6, 1e-5, 5e-4, 1e-4 | 5 |

**Total runs:** 4 × 5 × 5 × 5 = **500 simulations**  
**After filtering** Re_p > 300 (turbulent regime, outside Ergun validity):
expected **~300 usable points**.

### Variables Extracted Per Run

| Column | LBM extraction |
|---|---|
| `dP_L` | body force × ρ  (or Δρ · c_s² / L) |
| `v` | total x-flux / domain volume (superficial velocity) |
| `mu` | ν · ρ,  where ν = (τ − 0.5) / 3 (lattice units) |
| `rho` | domain-mean lattice density |
| `d` | 2 × sphere radius (in lattice units) |
| `phi` | n_fluid_nodes / n_total_nodes |

Output variable: `v` (velocity as a function of driving force and geometry).

### Quality Checks

- Discard runs with Re_p > 300 (beyond Ergun validity)
- Discard runs where the simulation did not reach steady state
  (check: |Δv/v| < 1e-5 between last two measurement intervals)
- Verify Ergun prediction vs LBM result: expect R² > 0.98 on the
  `f · φ³/(1-φ)` vs `Re_p/(1-φ)` collapse before running the pipeline

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi Reduction)

**Input:** The 3×6 dimension matrix above.

**What it does:**
1. Computes the null space via `pydimension.data_preprocessing.DataPreprocessor`
   with an explicit `dimension_matrix.csv`
2. Recovers the 3 primitive integer Pi-group basis vectors
3. Appends `log10` of each Pi group as reduced candidates for Step 2

**Expected output — 3 Pi groups:**
```
π1 = phi                           (porosity)
π2 = rho · v · d / mu             (Re_p)
π3 = dP_L · d · rho⁻¹ · v⁻²      (friction factor f)
```

**Verification:** known Ergun exponents projected onto null-space → cos = +1.0000

---

### Step 1 — Normalisation

**Input:** Raw physical X (n × 6) and output y (n,).

- **Min-max scaling** of raw X → `X_norm_raw` (for Step 3)
- **Min-max scaling** of 3 Pi features → `X_norm_step2` (for Step 2, pi-only mode)

---

### Step 2 — Latent Dimension Discovery

**Goal:** Find the smallest `k*` that can predict flow velocity.

**Input:** 3 Pi features (pi-only mode).

**Encoder architecture:** Multilayer MLP (hidden dims `[64, 32]`):
```
3  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```

Sweep over `k = 1, 2, 3`.

**Expected results:**

| k | Expected R² (test) |
|---|---|
| 1 | ~0.90 (Re alone, no φ dependence) |
| **2** | **~0.99 ← optimal** |
| 3 | overfitting |

**`k* = 2`** — both Re_p and φ are needed to collapse the data. The clean LBM
signal should give test R² near **0.99**, far exceeding the LPBF result of 0.77.

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or rotational.

**Input:** `X_norm_raw` — the 6 raw physical variables (always used here).

| Encoder | Transform | What it tests |
|---|---|---|
| Scaling | `z = W · log\|X\|` | Power-law / dimensional symmetry |
| Translational | `z = W · X` | Additive / affine symmetry |
| Rotational | `z = W · X²` | Quadratic / Euclidean symmetry |

**Expected results:**
```
scaling        : ~0.005   ← winner
translational  : ~0.05
rotational     : ~0.08
Loss gap: ≥ 10×
```

**Scaling wins** with a large gap — much cleaner than the LPBF result (1.2×)
or keyhole result (6.3×) because LBM data is exact and all variables vary
continuously.

---

### Step 4 — Generator Extraction

**Goal:** Extract directions in log-variable-space along which velocity is invariant.

**Input:** Winning encoder weight matrix `W` (shape `2 × 6`).

With `k* = 2` there are `6 − 2 = 4` null-space generators.

---

### Step 5 — Physical Interpretation

**Predicted generators:**

| Generator | Dominant trade-off | Physical meaning |
|---|---|---|
| G1 | ρ ↑, μ ↑ (same ratio) | Fluid swap at fixed Re — denser, more viscous fluid preserves flow |
| G2 | d ↑, v ↓ (same ratio) | Length-scale rescaling at fixed Re_p |
| G3 | dP_L ↑, ρ ↑, v² ↑ | Friction-factor preservation (inertial regime) |
| G4 | combined Re/f trade-off | Darcy-regime invariance |

**Key validations:**
- **Darcy invariance:** at fixed (φ, d), `v / dP_L = k / μ` — generator with
  weights `(+1, −1, +1, 0, 0, 0)` on (dP_L, v, mu, rho, d, phi)
- **Kozeny–Carman:** project exponents `(0, 0, 0, 0, 2, 3)` (d², φ³ → k)
  onto discovered null space → expect cos ≈ +1

---

## Output Files

All outputs go to `output_porous_media_lbm_symmetry/`.

| File | Contents |
|---|---|
| `dataset_lbm_porous.csv` | Generated dataset (~300 rows, 6 physics columns + Re/f reference) |
| `lbm_pi_candidates.png` | Pi-basis exponent heatmap + scatter of v vs each log₁₀(Πₖ) |
| `lbm_symmetry_discovery.png` | 2-panel: symmetry-type bar chart + latent-dim R² curve |
| `lbm_ergun_collapse.png` | Universal Ergun collapse: f·φ³/(1-φ) vs Re_p/(1-φ), all LBM points |
| `_da_repo/dimension_matrix.csv` | Explicit dimension matrix |
| `_da_repo/basis_vectors.csv` | Integer Pi-group exponent vectors |

---

## Expected Summary of Results

| Aspect | Expected Result |
|---|---|
| Pi groups discovered | 3 (φ, Re_p, friction factor f) |
| Known Ergun Pi in null-space | cos = +1.0000 ✓ |
| Latent dimension k* | **2** — Re_p and φ both needed |
| Test R² | **≥ 0.99** (noise-free LBM data) |
| Symmetry type | **Scaling** — ≥ 10× loss gap |
| Generators | 4 directions (6 variables − 2 latent dimensions) |
| All generators constrained? | **Yes** — all 6 variables vary continuously |

Contrast with LPBF:
- LPBF had 5-alloy confounding → 4 of 7 generators were under-determined
- LBM porous media: **all 4 generators fully determined** by the sweep design

---

## File Organisation

```
PyDimension/
├── pydimension/
└── projects/20260912_Stage1_Prokash/
    └── Examples/porous_media_lbm_symmetry/
        ├── PLAN.md                      ← this file
        ├── generate_dataset.py          ← LBM sweep script (TODO)
        ├── discover_symmetry.py         ← Stage1 pipeline script (TODO)
        └── output_porous_media_lbm_symmetry/
```

---

## Next Steps

1. **Share LBM code interface** — function signature for running one simulation
   (inputs: geometry, τ, body force; outputs: steady-state v, ρ, etc.)
2. **Write `generate_dataset.py`** — wraps the LBM code, runs the parameter
   sweep, saves `dataset_lbm_porous.csv`
3. **Write `discover_symmetry.py`** — Stage1 pipeline adapted for 6-variable
   porous media physics (follows `lpbf_porosity_symmetry/discover_symmetry.py`)
4. **Run and validate** — verify Ergun collapse before Stage1, then run full
   symmetry discovery

---

## Dependencies

`torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.  
Install with `pip install -r requirements.txt` from the repo root.
