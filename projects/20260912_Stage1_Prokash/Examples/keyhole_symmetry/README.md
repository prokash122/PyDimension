# Keyhole Welding Symmetry Discovery

Discover the hidden scaling symmetry in laser keyhole welding data using the
PyDimension Stage1 pipeline. No formula for the dimensionless groups is
supplied — the pipeline recovers the invariance entirely from the data.

---

## Physics Background

A focused laser beam drills a vapour cavity (keyhole) into a metal workpiece.
The keyhole eccentricity `e*` (also written `Ke`) depends on **seven physical
inputs** with **four fundamental dimensions** (Mass, Length, Time, Temperature):

| Variable | Symbol | SI Units | Dimensions |
|---|---|---|---|
| Absorbed laser power | `etaP` | W | kg·m²·s⁻³ |
| Scan speed | `Vs` | m/s | m·s⁻¹ |
| Beam radius | `r0` | m | m |
| Thermal diffusivity | `alpha` | m²/s | m²·s⁻¹ |
| Density | `rho` | kg/m³ | kg·m⁻³ |
| Specific heat capacity | `cp` | J/(kg·K) | m²·s⁻²·K⁻¹ |
| Superheat (liquidus − ambient) | `Tl-T0` | K | K |

By the **Buckingham Pi theorem** (7 variables − 4 dimensions = **3 independent
dimensionless groups**), eccentricity is fully determined by three Pi groups:

```
e* = f(π₁, π₂, π₃)
```

---

## Dataset — `dataset_keyhole.csv`

| Property | Value |
|---|---|
| Samples | 90 experimental points |
| Input columns | `etaP`, `Vs`, `r0`, `alpha`, `rho`, `cp`, `Tl-T0` |
| Output column | `e*` (keyhole eccentricity, dimensionless) |
| `Ke` range | 1.543 – 37.74 |

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi Reduction)

**Input:** The 7×4 dimension matrix (7 variables, 4 fundamental dimensions).

**What it does:**
Calls `pydimension.data_preprocessing.DataPreprocessor` to compute the
null-space of the dimension matrix, then simplifies the basis to **primitive
integer exponent vectors** via SymPy. An explicit `dimension_matrix.csv` is
written to `_da_repo/` and passed to the pipeline to bypass the unit-string
parser. Falls back to an inline scipy/SymPy implementation if pydimension is
unavailable.

**Actual output — 3 Pi groups discovered:**

```
π1 = Vs × r0 × alpha⁻¹          (Péclet number: advection / diffusion)
π2 = etaP × Vs⁻³ × r0⁻² × rho⁻¹  (power / kinetic energy flux)
π3 = Vs² × cp⁻¹ × (Tl-T0)⁻¹      (kinetic / thermal energy ratio)
```

Verification: the known Ke exponent vector projects onto the null-space with
**cos = +1.0000** — confirming the textbook formula lies exactly in the
discovered span.

---

### Step 1 — Normalisation

**Input:** Raw physical matrix `X` (90×7) and output `y` (90,).

**What it does:**
- Applies **min-max scaling** to raw physical `X` → `X_norm_raw` (for Step 3).
- Separately min-max scales the 3 Pi features → `X_norm_step2` (for Step 2).

**Outputs:**
- `X_norm_step2` — 3 Pi features scaled to [0, 1] (input to Step 2).
- `X_norm_raw` — 7 physical variables scaled to [0, 1] (input to Step 3 always).
- `y_norm` — min-max scaled eccentricity.

---

### Step 2 — Latent Dimension Discovery

**Goal:** Find the smallest number of coordinates `k*` that can predict `e*`.

**Input:** The **3 Pi features only** (pi-only mode, default). No `[X, X²,
log|X|]` augmentation — the encoder receives the dimensionless groups directly.

**Encoder architecture:** Multilayer MLP (default hidden dims `[64, 32]`):
```
3  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```
Paired with a nonlinear decoder (two 64-unit hidden layers). Trained end-to-end
to minimise MSE of `e*`. Sweep over `k = 1, 2, 3, 4`.

**Actual results:**

| k | R2_train | R2_test | MSE |
|---|---|---|---|
| **1** | **0.9821** | **0.9820** | **0.001108** ← optimal |
| 2 | 0.9853 | 0.9795 | 0.001263 |
| 3 | 0.9840 | 0.9797 | 0.001247 |
| 4 | 0.9840 | 0.9794 | 0.001270 |

**`k* = 1`** — a single latent coordinate explains 98.2% of the variance in
both train and test sets. Adding more latent dimensions gives negligible gain
(≤ 0.003 R² improvement) while the test R² slightly drops, indicating one
coordinate is sufficient. The near-perfect train/test R² alignment (0.9821 vs
0.9820) confirms no overfitting.

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or
rotational.

**Input:** Always `X_norm_raw` — the 7 raw physical variables.

> Raw physical X is always used here (not the Pi features) because the three
> competing encoders apply specific transforms: `X` (translational), `X²`
> (rotational), `log|X|` (scaling). These only make physical sense on
> multiplicatively-structured raw variables. Feeding pre-log-scaled Pi groups
> would produce `log(log(·))`, which is degenerate near zero.

**What it does:** Trains three single-linear-layer encoders, each applying a
different input transform before the shared decoder from Step 2:

| Encoder | Transform | What it tests |
|---|---|---|
| Scaling | `z = W · log|X|` | Power-law / dimensional symmetry |
| Translational | `z = W · X` | Additive / affine symmetry |
| Rotational | `z = W · X²` | Quadratic / Euclidean symmetry |

**Actual results:**

```
scaling        : 0.000982  ← winner
translational  : 0.006229
rotational     : 0.015462
Loss gap: 6.3×
```

**Scaling wins by 6.3×** over translational. This is a decisive result — the
keyhole physics obeys power-law dimensional relationships exactly as predicted
by Buckingham Pi, and the scaling encoder (`z = W · log|X|`) detects this with
far lower reconstruction error than the alternatives.

---

### Step 4 — Generator Extraction

**Goal:** Extract the explicit directions in variable space along which `e*` is
invariant.

**Input:** Winning (scaling) encoder weight matrix `W` (shape `1 × 7`).

**What it does:** For a scaling symmetry, invariance requires `W · g = 0`.
The generators are the **null-space of W** — there are `7 − 1 = 6` generators.

Each generator `g` is a 7-vector: simultaneously rescaling variable `i` by
`exp(ε·gᵢ)` for any `ε` leaves `e*` unchanged.

---

### Step 5 — Physical Interpretation

**Actual generators:**

| Generator | Trade-off | Physical meaning |
|---|---|---|
| 1 | increase `Vs` (+0.99), increase `etaP` (+0.14) | Faster scan + slightly more power preserves the energy-density balance |
| 2 | increase `r0` (+0.97), increase `etaP` (+0.20) | Larger beam radius compensated by more laser power |
| 3 | increase `alpha` (+0.72), decrease `etaP` (−0.63) | Higher thermal diffusivity drains heat faster; less laser power needed |
| 4 | increase `rho` (+0.85), decrease `etaP` (−0.46) | Higher-density metal absorbs more energy; less laser power needed |
| 5 | increase `cp` (+0.93), increase `etaP` (+0.32) | Higher specific heat compensated by more laser power |
| 6 | increase `Tl-T0` (+0.97), increase `etaP` (+0.22) | Higher superheat absorbed by more laser power |

With seven variables and `k* = 1`, the six generators span the entire
six-dimensional null space of `W`: any combination of these directions
keeps the keyhole eccentricity unchanged.

---

## Output Files

All outputs go to `output_keyhole_symmetry/` (configurable via `--output-dir`).

| File | Contents |
|---|---|
| `keyhole_pi_candidates.png` | Pi-basis exponent heatmap + scatter of `e*` vs each `log₁₀(Πₖ)` with logistic fit and R² |
| `keyhole_symmetry_discovery.png` | 3-panel: Pi-collapse, symmetry-type bar chart, discovered iso-invariant orbits in log-space |
| `_da_repo/dimension_matrix.csv` | Explicit dimension matrix fed to `DataPreprocessor` |
| `_da_repo/basis_vectors.csv` | Integer Pi-group exponent vectors |

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/keyhole_symmetry

# Default: Pi-only input to Step 2 (recommended)
python discover_symmetry.py --data dataset_keyhole.csv

# Deeper encoder:
python discover_symmetry.py --data dataset_keyhole.csv --encoder-hidden 128 64 32

# Ablation: disable pi-only — feed [X, X², log|X|, Pi] augmented input to Step 2
python discover_symmetry.py --data dataset_keyhole.csv --no-pi-only

# Increase training length for more stable results:
python discover_symmetry.py --data dataset_keyhole.csv --sym-epochs 2000
```

Default training budget: `--latent-epochs 600`, `--sym-epochs 1500`,
`--n-restarts 3`, `--encoder-hidden 64 32`.

---

## Summary of Observed Results

| Aspect | Result |
|---|---|
| Pi groups discovered | 3 (π1 = Péclet, π2 = power/kinetic, π3 = kinetic/thermal) |
| Known Ke in null-space | cos = +1.0000 ✓ |
| Latent dimension k* | **1** — single coordinate explains 98.2% variance |
| Test R² | **0.9820** (train and test nearly identical — no overfitting) |
| Symmetry type | **Scaling** — 6.3× loss gap over translational |
| Generators | 6 directions (7 variables − 1 latent dimension) |
| Constrained generators | etaP-Vs, etaP-r0, alpha-etaP, rho-etaP, cp-etaP, Tl-T0-etaP |
| Free generators | none (six generators span the full 6-D null space) |

---

## File Organization

Scripts auto-discover the `pydimension` package by walking upward from
`discover_symmetry.py`'s location — no installation needed if inside the repo:

```
PyDimension/
├── pydimension/                    ← auto-discovered
│   └── data_preprocessing/
└── projects/20260912_Stage1_Prokash/
    ├── preprocessing/
    ├── intrinsic_coordinate/
    ├── symmetry_discovery/
    └── Examples/keyhole_symmetry/
        ├── discover_symmetry.py
        └── dataset_keyhole.csv
```

Dependencies: `torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.
Install with `pip install -r requirements.txt` from the repo root.
