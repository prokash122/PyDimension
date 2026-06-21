# LPBF Porosity Scaling-Symmetry Discovery

Discover the hidden scaling symmetry in Laser Powder Bed Fusion (LPBF)
additive-manufacturing porosity data using the PyDimension Stage1 pipeline.
No formula for the dimensionless groups is supplied — the pipeline recovers
the invariance entirely from the experimental data.

---

## Motivation

Experimental porosity measurements for five alloys (Al2024, Al6061, Cu,
SS304, Ti64) at many (P, V) combinations can be collapsed onto a single
logistic curve using the hand-picked "normalised enthalpy":

```
Pi = (Lv · rho · A · P · V) / (k² · (Tb − Tm)²)
```

This collapse is *evidence* of a scaling symmetry — but the notebook that
demonstrates it never extracts the symmetry itself. This example closes that
gap: it recovers the invariance and the physical trade-offs directly from the
data, without knowing the formula for `Pi` in advance.

---

## Physics Background

The pore fraction `f` depends on **eleven physical inputs** with **four
fundamental dimensions** (Mass, Length, Time, Temperature):

| Variable | Symbol | SI Units | Dimensions |
|---|---|---|---|
| Laser power | `P` | W | kg·m²·s⁻³ |
| Scan speed | `V` | m/s | m·s⁻¹ |
| Absorptivity | `A` | — | dimensionless |
| Metal density | `rho` | kg/m³ | kg·m⁻³ |
| Thermal conductivity | `k` | W/(m·K) | kg·m·s⁻³·K⁻¹ |
| Latent heat of vapourisation | `Lv` | J/kg | m²·s⁻² |
| Superheat (Tb − Tm) | `dT` | K | K |
| Surface tension | `gamma` | N/m | kg·s⁻² |
| Boiling temperature | `Tb` | K | K |
| (Tb − Tm) — same value as `dT` | `Tb_minus_Tm` | K | K |
| (Tm − T0), T0 = 298 K | `Tm_minus_T0` | K | K |

By the **Buckingham Pi theorem** (11 variables − 4 dimensions = **7 independent
dimensionless groups**). These seven groups are the *only* features fed to
Step 2. The two extra temperature columns are added so the encoder has
explicit access to both factors of the Pe_vap denominator
`(Tb − Tm) · (Tm − T0)` (see Step 2b — both Pe_vap and Pr now end up
inside the latent span).

Two further dimensionless quantities from the keyhole-mode-transition
literature are computed per row and used **only** to validate the latent
space discovered in Step 2 (and, if absent from it, injected into Step 3 —
see Step 2b):

```
Pe_vap = (Lv · rho · A · P · V) / (k² · (Tb − Tm) · (Tm − T0))     vaporisation Péclet
Pr     = (η · Cp) / k                                              thermal Prandtl
```

`Tm`, `η`, and `Cp` are looked up per material from a built-in table
(`PRESSURE_PROPS`); `T0 = 298 K`.

> `gamma` and `Tb` are not in `dataset_lpbf.csv`. They are added at runtime
> from a built-in per-material table so the full 9-variable space is available.

---

## Dataset — `dataset_lpbf.csv`

| Property | Value |
|---|---|
| Rows | 232 experimental points |
| Alloys | Al2024, Al6061, Cu, SS304, Ti64 |
| Raw columns | `P`, `V`, `A`, `rho`, `k`, `Lv`, `dT`, `Pi` (reference), `Pore` |
| Output | `Pore` — pore fraction in [0, 1] |
| `Pi` column | Pre-computed reference only; **not used as model input** |
| Source | Built from `lpbf_porosity_dataset.xlsx` (sheet `pore fraction`); sample IDs `Al20XX`/`Al60XX`/`Cu`/`Ti`/`ss304` mapped to the five canonical alloys, with per-alloy `(A, ρ, k, L_v, ΔT, γ, T_b)` joined in at load time |

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi Reduction)

**Input:** The 9×4 dimension matrix (9 variables, 4 fundamental dimensions).

**What it does:**
1. Enriches the CSV at runtime by appending `gamma` and `Tb` columns (looked
   up from `PRESSURE_PROPS` by material name), writing
   `_da_repo/dataset_lpbf_enriched.csv`.
2. Calls `pydimension.data_preprocessing.DataPreprocessor` with an explicit
   `dimension_matrix.csv` (bypasses the unit-string parser which mishandles
   `W/(m·K)`). Falls back to an inline scipy/SymPy implementation if needed.

**Actual output — 5 Pi groups discovered:**

```
π1 = A                               (absorptivity is dimensionless → own group)
π2 = V² × Lv⁻¹                       (kinetic / latent heat ratio)
π3 = P × V³ × rho × k⁻² × dT⁻²      (normalised enthalpy form)
π4 = P × V × rho × gamma⁻²           (capillary group)
π5 = P × V³ × rho × k⁻² × Tb⁻²      (normalised enthalpy with Tb)
```

Verification: the known normalised-enthalpy exponent vector
`[1, 1, 1, 1, −2, 1, −2, 0, 0]` projects onto the null-space with
**cos = +1.0000** — confirming it lies exactly in the discovered span.

---

### Step 1 — Normalisation

**Input:** Raw physical matrix `X` (232×9) and output `y` (232,).

**What it does:**
- Applies **min-max scaling** to raw physical `X` → `X_norm_raw` (for Step 3).
- Separately min-max scales the 5 Pi features → `X_norm_step2` (for Step 2).

**Outputs:**
- `X_norm_step2` — 5 Pi features scaled to [0, 1] (input to Step 2).
- `X_norm_raw` — 9 physical variables scaled to [0, 1] (input to Step 3 always).
- `y_norm` — min-max scaled pore fraction.

---

### Step 2 — Latent Dimension Discovery

**Goal:** Find the smallest `k*` that can predict pore fraction.

**Input:** The **7 Pi features only** (pi-only mode, default).

> Pi-only is the recommended default for this dataset. The LPBF material-property
> columns (`A`, `rho`, `k`, `Lv`, `dT`) only take 5 discrete values (one per
> alloy) — after min-max scaling, entire material groups collapse to 0, and
> `log(0)` would produce NaN. The Pi features are always well-defined.

**Encoder architecture:** Multilayer MLP (default hidden dims `[64, 32]`):
```
7  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```
Paired with a nonlinear decoder (two 64-unit hidden layers). Sweep over
`k = 1, 2, 3, 4`.

**Actual results:**

| k | R2_train | R2_test | MSE |
|---|---|---|---|
| 1 | 0.8330 | 0.6739 | 0.032767 |
| 2 | 0.8813 | 0.7425 | 0.025876 |
| 3 | 0.8736 | 0.7352 | 0.026606 |
| **4** | **0.9059** | **0.7738** | **0.022733** ← optimal |

**`k* = 4`** — with two extra temperature columns and a 7-feature Pi space,
the optimal Step 2 bottleneck widens from `k=1` (the 5-feature setup) to
`k=4`. The 4-D latent now has enough room to encode both Pe_vap and the
thermal Prandtl Pr explicitly (Step 2b below).

---

### Step 2b — Does `z` Encode Pe_vap and Pr?

**Goal:** Check whether the two discovered latent coordinates already
contain `Pe_vap` and the thermal Prandtl number `Pr`. Whichever is missing
gets appended to the Step 3 input so the single-layer symmetry encoder can
see it directly.

**What it does:**
1. Evaluates the trained Step 2 encoder on every sample → `z` of shape
   `(232, k*)` (here `k* = 1`).
2. Fits two ordinary linear regressions, `z → log10(Pe_vap)` and
   `z → log10(Pr)`, and reports the R² of each.
3. Anything with `R² < 0.80` (the *discovered* threshold) is flagged as
   **not in the latent span** and injected into Step 3 — but **into the
   bottleneck, not the encoder input** (Option A).

   The Step 3 encoder still acts only on the 9 raw physical variables,
   producing a `k*`-dim projection `W·ϕ_s(X_9)`. Pe_vap and Pr are min-max
   scaled and **concatenated to the encoder output** before the decoder
   sees them. The decoder therefore receives `[W·ϕ_s(X_9), Pe_vap, Pr]`,
   so the Pi quantities are given to it for free — the symmetry-class
   encoder only has to explain whatever is left after Pe_vap and Pr are
   already accounted for.

**Actual results (with the two added temperature columns):**

| Quantity | R²(`z` → log10·) | Verdict |
|---|---|---|
| Pe_vap | **0.9971** | ✓ in latent span |
| Pr     | **0.9754** | ✓ in latent span |

→ Nothing is injected into Step 3. The 4-D Step 2 latent already encodes
both manuscript Pi quantities, so the symmetry-type encoder runs on the
**9 raw physical variables with no extras** (encoder input = 9, decoder
input = `k* = 4`).

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or
rotational.

**Encoder input:** the 9 raw physical variables (unchanged).
**Bottleneck:** `W·ϕ_s(X_9)` of dimension `k* = 4`. Nothing extra is
concatenated because Step 2b's diagnostic found Pe_vap and Pr already in
the latent span.

> Raw physical X is always used here (not the Pi features from Step 2)
> because the three competing encoders apply transforms `X`, `X²`, `log|X|`
> that are only physically meaningful on raw multiplicatively-structured
> variables. Feeding pre-log-scaled Pi groups would produce `log(log(·))` —
> degenerate near zero.

**What it does:** Trains three competing single-linear-layer encoders:

| Encoder | Transform | What it tests |
|---|---|---|
| Scaling | `z = W · log|X|` | Power-law / dimensional symmetry |
| Translational | `z = W · X` | Additive / affine symmetry |
| Rotational | `z = W · X²` | Quadratic / Euclidean symmetry |

**Actual results (k* = 4, no bottleneck extras):**

```
rotational     : 0.015570  ← winner
scaling        : 0.016160
translational  : 0.025359
Loss gap: 1.0×
```

**The winner flips to rotational, but with `gap = 1.0×` it is a tie**
between rotational and scaling. METHOD.md requires `gap > 3` for a
confident detection, so the conclusion here is "ambiguous between
rotational and scaling, with translational clearly behind."

Two compounding effects narrow the gap relative to the 9-variable run:
- `k* = 4` gives every class a much wider bottleneck (4 dims instead of 1).
  More capacity → all classes fit better and their losses converge.
- Adding `(Tb − Tm)` (a duplicate of `dT`) and `(Tm − T0)` enlarges the
  encoder's input space, again helping every class equally.

In other words, putting Pe_vap and Pr fully inside the latent span has the
side effect of dissolving the symmetry-class signal.

---

### Step 4 — Generator Extraction

**Goal:** Extract directions in log-variable-space along which pore fraction
is invariant.

**Input:** Winning encoder weight matrix `W` (shape `4 × 11` — the 11
physical variables, no extras).

**Encoder weight rows (L2-normalised):**

```
         P        V        A      rho        k       Lv       dT    gamma       Tb  Tb-Tm   Tm-T0
Row 1: +0.224  +0.600  +0.182  -0.257  -0.201  -0.301  +0.106  +0.209  -0.002  -0.407  +0.374
Row 2: -0.499  -0.212  +0.300  +0.378  +0.129  +0.153  +0.484  +0.176  +0.053  +0.387  +0.124
Row 3: +0.586  +0.335  +0.263  -0.435  +0.363  +0.041  +0.116  +0.011  -0.247  +0.172  -0.220
Row 4: -0.242  +0.526  +0.330  -0.229  -0.232  -0.407  +0.146  -0.153  +0.242  -0.409  +0.120
```

`cos<row, known-Pi-exponents>` per row: `+0.18, −0.31, −0.05, +0.04` — no
single row aligns cleanly with the textbook normalised-enthalpy direction,
which is expected for a winning *rotational* encoder (the symmetry class
makes the rows quadratic-invariants rather than scaling-Pi exponents).

With `k* = 4` and an 11-D encoder input the rotational class produces
**45 antisymmetric generators** by clustering equal-weight indices into
pairs across the 4 latent rows.

---

### Step 5 — Physical Interpretation

The rotational class emits **45 antisymmetric generators** here — one
per pair of indices within each equal-weight cluster across the 4 latent
rows. Per-generator descriptions are too numerous to tabulate in the
README; see `run.log` for the full list.

A key qualitative point: rotational generators describe `(i, j)` index
*rotations* `x → exp(ε A_{ij}) x` in input space rather than the
multiplicative trade-offs that the scaling generators describe. Because
the gap to scaling is essentially zero (1.0×), neither rotational nor
scaling generators should be over-interpreted here — the cleaner
generator stories live in the simpler 9-variable / `k* = 1` runs.

**Caveat — 5-alloy confounding still applies.** Eight of the eleven
encoder columns (`A, rho, k, Lv, dT, gamma, Tb, Tb_minus_Tm, Tm_minus_T0`)
take at most 5 distinct values across the dataset (one per alloy), so any
generator dominated by them is data-limited. Only the `P` and `V`
directions are continuously varied within a material.

---

## Output Files

All outputs go to `output_lpbf_porosity_symmetry/` (configurable via
`--output-dir`).

| File | Contents |
|---|---|
| `lpbf_pi_candidates.png` | Pi-basis exponent heatmap + scatter of pore fraction vs each `log₁₀(Πₖ)` with logistic fit and R² |
| `lpbf_porosity_symmetry_discovery.png` | 3-panel: Pi-collapse, symmetry-type bar chart, discovered iso-invariant orbits in (log V, log P) |
| `_da_repo/dataset_lpbf_enriched.csv` | CSV enriched with `gamma`, `Tb` columns |
| `_da_repo/dimension_matrix.csv` | Explicit dimension matrix fed to `DataPreprocessor` |
| `_da_repo/basis_vectors.csv` | Integer Pi-group exponent vectors |

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/lpbf_porosity_symmetry

# Default: Pi-only input to Step 2 (recommended)
python discover_symmetry.py --data dataset_lpbf.csv

# Deeper encoder:
python discover_symmetry.py --data dataset_lpbf.csv --encoder-hidden 128 64 32

# Ablation: disable pi-only — feed [X, X², log|X|, Pi] augmented input to Step 2
python discover_symmetry.py --data dataset_lpbf.csv --no-pi-only

# Log-prenormalisation (geometric-mean centring before min-max):
python discover_symmetry.py --data dataset_lpbf.csv --log-normalize

# Increase training length for more stable symmetry detection:
python discover_symmetry.py --data dataset_lpbf.csv --sym-epochs 2000
```

Default training budget: `--latent-epochs 600`, `--sym-epochs 1500`,
`--n-restarts 3`, `--encoder-hidden 64 32`.

---

## Summary of Observed Results

| Aspect | Result |
|---|---|
| Pi groups discovered | 5 (A, V²/Lv, normalised-enthalpy form, capillary, Tb-variant) |
| Known Pi in null-space | cos = +1.0000 ✓ |
| Latent dimension k* | **1** — a single coordinate suffices for pore fraction |
| Test R² | **0.777** (vs ~0.446 from raw Pi formula) |
| Pe_vap / Pr in latent span? | R² = 0.997 / 0.975 → **both already discovered** |
| Injection style | **None** — nothing added to Step 3 |
| Step 3 encoder input | 11 raw physical variables |
| Step 3 bottleneck dim | **4** (`k* = 4`, no extras) |
| Symmetry type | **Rotational** at 0.0156 vs **Scaling** at 0.0162 — **1.0× gap (tied)** |
| Generators | **45** antisymmetric pairs (rotational, 4 latent rows × clustered indices) |
| Caveat | 5-alloy confounding still limits any material-only direction |

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
    └── Examples/lpbf_porosity_symmetry/
        ├── discover_symmetry.py
        └── dataset_lpbf.csv
```

Dependencies: `torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.
Install with `pip install -r requirements.txt` from the repo root.
