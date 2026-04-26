# Keyhole Welding Symmetry Discovery

Discover the hidden scaling symmetry in laser keyhole welding data using the
PyDimension Stage1 pipeline. The pipeline works entirely from raw physical
measurements — no formula for the dimensionless groups is supplied.

---

## Physics Background

A focused laser beam drills a vapour cavity (keyhole) into a metal workpiece.
The keyhole eccentricity `e*` (also written `Ke`) is the single scalar output
that characterises keyhole shape across many experiments.

It depends on **seven physical inputs** with **four fundamental dimensions**
(Mass, Length, Time, Temperature):

| Variable | Symbol | SI Units | Dimensions |
|---|---|---|---|
| Absorbed laser power | `etaP` | W | kg·m²·s⁻³ |
| Scan speed | `Vs` | m/s | m·s⁻¹ |
| Beam radius | `r0` | m | m |
| Thermal diffusivity | `alpha` | m²/s | m²·s⁻¹ |
| Density | `rho` | kg/m³ | kg·m⁻³ |
| Specific heat capacity | `cp` | J/(kg·K) | m²·s⁻²·K⁻¹ |
| Superheat (liquidus − ambient) | `Tl-T0` | K | K |

By the **Buckingham Pi theorem** (7 variables − 4 dimensions = 3 groups),
`e*` is controlled by exactly **3 independent dimensionless groups**:

```
e* = f(π₁, π₂, π₃)
```

One known combination is the **Péclet number** `Pe = Vs·r0 / alpha`, which
characterises the ratio of advective to diffusive heat transport. The Stage1
pipeline recovers all three groups and their associated scaling symmetry
directly from the experimental data.

---

## Dataset — `dataset_keyhole.csv`

| Property | Value |
|---|---|
| Columns | `etaP`, `Vs`, `r0`, `alpha`, `rho`, `cp`, `Tl-T0`, `e*` |
| Output | `e*` (keyhole eccentricity, dimensionless) |

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi Reduction)

**Input:** The 7×4 dimension matrix (7 variables, 4 fundamental dimensions).

**What it does:**

1. Computes the null-space of the dimension matrix to find all independent
   dimensionless combinations.
2. Simplifies the null-space basis to **primitive integer exponent vectors**
   via SymPy (e.g. `[1, −1, 0, −1, 0, 0, 0]` instead of
   `[0.5, −0.5, 0, −0.5, 0, 0, 0]`).
3. Uses `pydimension.data_preprocessing.DataPreprocessor` (the repo's
   built-in pipeline) when available; falls back to an inline scipy
   implementation otherwise.
4. Verifies that the known Péclet-like exponent vector lies in the
   null-space span (cosine similarity printed at runtime).

**Output:** `pi_basis` — a 7×3 integer matrix whose columns are the three
Pi-group exponent vectors, i.e. the exponents `[α₁, α₂, …, α₇]` such that
`Πₖ = etaP^α₁ · Vs^α₂ · … · (Tl-T0)^α₇`.

Also written to disk: `output_keyhole_symmetry/_da_repo/dimension_matrix.csv`
and `basis_vectors.csv`.

---

### Step 1 — Normalisation

**Input:** Raw physical matrix `X` (N×7) and output vector `y` (N).

**What it does:**

- Applies **min-max scaling** to each column of `X` so every variable lies
  in `[0, 1]`. This is always done on the raw physical variables.
- In `--pi-only` mode the Pi features are separately min-max scaled for
  use in Step 2; the raw physical `X` is always kept for Step 3.

**Outputs:**
- `X_norm_raw` — min-max scaled physical variables (used in Step 3).
- `X_norm_step2` — input for Step 2: either `X_norm_raw` (default) or the
  min-max scaled Pi features (with `--pi-only`).
- `y_norm` — min-max scaled output.

---

### Step 2 — Latent Dimension Discovery

**Goal:** Find the smallest number of coordinates `k*` that can predict `e*`.

**Input (default mode):** `X_norm_step2` augmented with
`[X, X², log|X|, π₁, π₂, π₃]` — a wide feature vector that gives the
encoder access to both the raw physical variables and the precomputed
dimensionless groups. The augmentation is handled internally by
`discover_latent_dimension`.

**Input (`--pi-only` mode):** Only the 3 Pi features (no `[X, X², log|X|]`
augmentation). The encoder receives dimensionless groups directly.

**Encoder architecture:** A **multilayer MLP** (default hidden dims
`[64, 32]`, configurable via `--encoder-hidden`):

```
input_dim  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```

The encoder is paired with a fixed nonlinear decoder (two hidden layers of
width 64) and trained end-to-end to minimise reconstruction MSE of `y`.

A sweep over `k = 1, 2, 3, 4` is run; the optimal `k*` is chosen by the
elbow in the test R² curve.

**Output:**
- `optimal_n_latent` (`k*`) — the chosen intrinsic dimension.
- `best_encoder`, `best_decoder` — trained PyTorch modules.
- Per-k metrics: `R2_train`, `R2_test`, `MSE`.

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is **scaling**, **translational**,
or **rotational**.

**Input:** Always `X_norm_raw` (raw physical variables, never Pi features).

> **Why raw X?** The three symmetry-type encoders apply specific transforms
> to the input — `X` (translational), `X²` (quadratic / rotational), and
> `log|X|` (scaling). These transforms are only physically meaningful on
> multiplicatively-structured raw variables. Feeding pre-log-scaled Pi groups
> would produce `log(log(·))` — undefined near zero and not informative.

**What it does:** Trains three competing single-linear-layer encoders, each
applying a different input transform before the same shared decoder (reused
from Step 2):

| Encoder | Transform | Hypothesis |
|---|---|---|
| Scaling | `z = W · log|X|` | power-law / dimensional invariance |
| Translational | `z = W · X` | additive / affine invariance |
| Rotational | `z = W · X²` | quadratic / Euclidean invariance |

The encoder with the **lowest reconstruction loss** wins. The loss ratio
(second-best / best) quantifies the confidence of the detection.

**Output:**
- `symmetry_type` — winning type (`"scaling"`, `"translational"`, or
  `"rotational"`).
- `losses` — dict of final losses for all three types.
- `encoders` — trained encoder module for each type.

---

### Step 4 — Generator Extraction

**Goal:** Extract the explicit directions in variable space along which the
output is invariant.

**Input:** The winning encoder's weight matrix `W` (shape `k* × 7`).

**What it does:** For a scaling symmetry, the output is invariant to
simultaneous rescaling `xᵢ → xᵢ · exp(εgᵢ)` for any scalar `ε`.
The encoder computes `z = W · log|X|`, so invariance requires `W · g = 0`.
The generators are therefore the **null-space of W**:

```
generators = null_space(W)   # shape: (7, 7 − k*)
```

Each generator is a 7-dimensional vector whose entries are the **exponents**
of the simultaneous rescaling. There are `7 − k* = 4` to `6` generators
depending on `k*`.

**Output:** List of generator vectors (each length 7, one entry per variable
in `VARIABLE_NAMES` order).

---

### Step 5 — Physical Interpretation

**What it does:** For each generator `g`:

- Lists which variables have a non-negligible coefficient (`|gᵢ| > 0.05`).
- Identifies the dominant variable (largest `|gᵢ|`).
- States the trade-off: e.g. "increase Vs while decrease r0 to keep Ke
  (and e*) unchanged".

---

## Output Files

All outputs are written to `output_keyhole_symmetry/` (configurable via
`--output-dir`).

| File | Contents |
|---|---|
| `keyhole_pi_candidates.png` | Pi-basis heatmap + scatter of `e*` vs each `log₁₀(Πₖ)` with logistic fit and R² |
| `keyhole_symmetry_discovery.png` | 3-panel summary: Pi-collapse, symmetry-type bar chart, discovered iso-invariant orbits in log-space |
| `_da_repo/dimension_matrix.csv` | Explicit dimension matrix fed to `DataPreprocessor` |
| `_da_repo/basis_vectors.csv` | Integer Pi-group exponent vectors from `DataPreprocessor` |

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/keyhole_symmetry

# Default: repo DA pipeline + multilayer encoder [64, 32] + Pi features injected
python discover_symmetry.py --data dataset_keyhole.csv

# Deeper encoder:
python discover_symmetry.py --data dataset_keyhole.csv --encoder-hidden 128 64 32

# Pi-only mode: Step 2 uses only the 3 Pi groups as input; Step 3 uses raw X
python discover_symmetry.py --data dataset_keyhole.csv --pi-only

# Ablation: no Pi injection into Step 2 (raw X + augmentation only):
python discover_symmetry.py --data dataset_keyhole.csv --no-pi-input

# Increase training length for more stable symmetry detection:
python discover_symmetry.py --data dataset_keyhole.csv --sym-epochs 2000
```

Default training budget: `--latent-epochs 600`, `--sym-epochs 1500`,
`--n-restarts 3`.

---

## Expected and Observed Results

### Step 0 — Dimensional Analysis

**Expected:** 3 integer Pi groups from a rank-4 dimension matrix.

One representative basis (up to column permutation and integer scaling):

```
Pi1 = etaP^1 · Vs^-1 · r0^-2 · alpha^0 · rho^-1 · cp^-1 · (Tl-T0)^-1
Pi2 = etaP^0 · Vs^1 · r0^1 · alpha^-1   (Péclet number Pe)
Pi3 = etaP^0 · Vs^0 · r0^1 · alpha^0 · rho^1 · cp^1 · (Tl-T0)^1
```

Cosine similarity of the known Ke exponents `[1, −0.5, −1.5, −0.5, −1, −1, −1]`
against their projection onto the discovered null-space ≈ **±1.00**, confirming
the known formula lies exactly in the discovered span.

---

### Step 2 — Latent Dimension

**Expected:** `k* = 2` or `3` (three Pi groups control eccentricity, but some
may correlate strongly enough to collapse to two).

| k | R2_train | R2_test | MSE |
|---|---|---|---|
| 1 | moderate | moderate | higher |
| 2 | high | **best** | low |
| 3 | high | similar to k=2 | low |

`k* = 2` or `k* = 3` — eccentricity is well described by two to three latent
coordinates. The multilayer encoder (hidden dims `[64, 32]`) gives
substantially better R² than a single linear projection.

---

### Step 3 — Symmetry Type

**Expected:** Scaling wins by a large margin.

```
scaling       : 0.XXXX  ← winner
translational : 0.XXXX
rotational    : 0.XXXX
Loss gap: ~5–6×
```

Scaling wins decisively (5.8× gap observed in testing) because the data
follows power-law dimensional relationships — exactly what the scaling
encoder (`z = W · log|X|`) is designed to detect.

---

### Step 4 — Generator Directions

With `k* = 2`, there are `7 − 2 = 5` null-space generators.

**Constrained generators** (meaningful physical trade-offs):
- **Péclet trade-off**: increase `Vs`, decrease `r0` → preserving `Pe = Vs·r0/alpha`.
- **Power–diffusivity trade-off**: increase `etaP`, decrease `alpha` (and
  compensating `rho`, `cp`, `Tl-T0` terms).

**Free generators** (under-determined directions): variables that do not
vary independently across the dataset contribute near-zero encoder weights,
and their generators are nearly unconstrained.

---

### Summary

| Aspect | Result |
|---|---|
| Symmetry type | **Scaling** (5.8× loss gap) |
| Latent dimension | k* = 2–3 |
| Known Ke in null-space | cos ≈ ±1.00 ✓ |
| Péclet trade-off | Recovered ✓ |
| Underdetermined variables | Reported honestly |

---

## File Organization

The scripts auto-discover the `pydimension` package by walking upward from
`discover_symmetry.py`'s location. No installation or path configuration is
needed as long as the scripts remain inside the repo tree:

```
PyDimension/                              ← repo root
├── pydimension/                          ← found automatically
│   └── data_preprocessing/
└── projects/20260912_Stage1_Prokash/
    ├── preprocessing/
    ├── intrinsic_coordinate/
    ├── symmetry_discovery/
    └── Examples/
        └── keyhole_symmetry/
            ├── discover_symmetry.py
            └── dataset_keyhole.csv
```

Dependencies: `torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.
Install with `pip install -r requirements.txt` from the repo root.
