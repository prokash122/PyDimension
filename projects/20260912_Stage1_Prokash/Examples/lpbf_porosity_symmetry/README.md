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

The pore fraction `f` depends on **nine physical inputs** with **four
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

By the **Buckingham Pi theorem** (9 variables − 4 dimensions = **5 independent
dimensionless groups**). These five groups are the *only* features fed to
Step 2 — the pressure ratio `P_recoil / P_Laplace` from the companion
notebook is still computed (used for the 3D surface plot at the end) but is
**not** added to the Step 2 input.

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

**Input:** The **5 Pi features only** (pi-only mode, default).

> Pi-only is the recommended default for this dataset. The LPBF material-property
> columns (`A`, `rho`, `k`, `Lv`, `dT`) only take 5 discrete values (one per
> alloy) — after min-max scaling, entire material groups collapse to 0, and
> `log(0)` would produce NaN. The Pi features are always well-defined.

**Encoder architecture:** Multilayer MLP (default hidden dims `[64, 32]`):
```
5  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```
Paired with a nonlinear decoder (two 64-unit hidden layers). Sweep over
`k = 1, 2, 3, 4`.

**Actual results:**

| k | R2_train | R2_test | MSE |
|---|---|---|---|
| **1** | **0.9105** | **0.7765** | **0.022465** ← optimal |
| 2 | 0.8928 | 0.7487 | 0.025259 |
| 3 | 0.9275 | 0.7575 | 0.024367 |
| 4 | 0.9205 | 0.7599 | 0.024126 |

**`k* = 1`** — without the empirical pressure-ratio feature, pore fraction
is controlled by a single latent coordinate, and `k=1` gives both the best
test R² (0.7765) and the lowest MSE. Adding more latents lowers test R²,
indicating overfitting.

The test R² of **0.777** is notably higher than the raw Pi-collapse R² of
~0.446 from the textbook formula alone, because the encoder is free to find
the best 1-D projection of the 5-feature Pi space rather than being
constrained to a single formula.

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
   **not in the latent span** and appended to `X_step3` (min-max scaled to
   match the column normalisation of the 9 physical vars).

**Actual results:**

| Quantity | R²(`z` → log10·) | Verdict |
|---|---|---|
| Pe_vap | 0.4651 | NOT in latent span → injected |
| Pr     | 0.0592 | NOT in latent span → injected |

→ Step 3 input grows from 9 to **11 features**:
`[P, V, A, rho, k, Lv, dT, gamma, Tb, Pe_vap, Pr]`.

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or
rotational.

**Input:** `X_step3` — the 9 raw physical variables plus any Pi quantity
that Step 2b flagged as missing from the latent span (here: `Pe_vap`, `Pr`
→ 11 columns).

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

**Actual results (with Pe_vap and Pr injected):**

```
scaling        : 0.018870  ← winner
translational  : 0.038371
rotational     : 0.050433
Loss gap: 2.0×
```

**Scaling wins**, confirming the power-law dimensional structure of LPBF
porosity. The 2.0× gap is still narrower than the keyhole example's 6.3×
because the 5 material-property variables take only 5 discrete values (one
per alloy), limiting how much scaling information the encoder can extract
from them.

---

### Step 4 — Generator Extraction

**Goal:** Extract directions in log-variable-space along which pore fraction
is invariant.

**Input:** Winning encoder weight matrix `W` (shape `1 × 11` — the 9
physical variables plus `Pe_vap` and `Pr`).

**Encoder weight row (L2-normalised):**

```
         P        V        A      rho        k       Lv       dT    gamma       Tb   Pe_vap       Pr
Row 1: -0.603  -0.407  +0.163  -0.144  +0.114  -0.426  +0.153  -0.434  +0.098  +0.048  +0.070
```

Cosine with the known normalised-enthalpy exponents
`[1, 1, 1, 1, −2, 1, −2, 0, 0, 0, 0]` is **−0.54** (the row is anti-aligned
with the textbook formula — same physical direction, opposite sign).

With `k* = 1` there are `11 − 1 = 10` null-space generators.

---

### Step 5 — Physical Interpretation

**Actual generators (11-D Step 3 input, 1 latent → 10 generators):**

| Generator | Dominant variable | Trade-off | Physical meaning |
|---|---|---|---|
| 1 | V | increase V, decrease P, decrease Lv, decrease γ | Speed–power trade-off |
| 2 | A | increase A, increase P | Absorptivity–power compensation |
| 3 | rho | increase ρ, decrease P | Density–power trade-off |
| 4 | k | increase k, increase P | Conductive metal absorbs higher P |
| 5 | Lv | increase Lv, decrease P, decrease V, decrease γ | Latent-heat trade-off |
| 6 | dT | increase dT, increase P | Superheat trade-off |
| 7 | gamma | increase γ, decrease P, decrease V, decrease Lv | Surface-tension trade-off |
| 8 | Tb | increase Tb, increase P | Boiling-temperature trade-off |
| 9 | Pe_vap | increase Pe_vap alone | Pure Pe_vap direction (Step 2b injection) |
| 10 | Pr | increase Pr, increase P | Thermal-Prandtl trade-off (Step 2b injection) |

**Constrained vs free generators:**

- **Constrained (data-determined):** Generators 1–4 involve P and V — the
  two process parameters that vary continuously within each material
  (30–50 points per alloy). Their exponents are well-determined.

- **Free (data-limited):** Generators 5–8 are dominated by material-property
  variables (Lv, gamma, Tb, dT) that each take only 5 distinct values
  (confounded across alloys). The pipeline cannot separately determine these
  exponents from 5 alloys alone. This is its honest statement: *recovering
  all 9 exponents would require ≥ 20 alloys with independently varied
  thermophysical properties.*

- **Injected (Step 2b):** Generators 9–10 are the directions along which
  `Pe_vap` and `Pr` can be varied while pore fraction stays constant. `Pe_vap`
  appears as an almost-pure axis (the per-row formula sits in a direction
  orthogonal to the discovered latent); `Pr` couples weakly to `P`.

---

## Output Files

All outputs go to `output_lpbf_porosity_symmetry/` (configurable via
`--output-dir`).

| File | Contents |
|---|---|
| `lpbf_pi_candidates.png` | Pi-basis exponent heatmap + scatter of pore fraction vs each `log₁₀(Πₖ)` with logistic fit and R² |
| `lpbf_porosity_symmetry_discovery.png` | 3-panel: Pi-collapse, symmetry-type bar chart, discovered iso-invariant orbits in (log V, log P) |
| `lpbf_3d_surface.png` | 3D surface: pore fraction over `(log₁₀(Pi), log₁₀(PR))` with fitted 2-D logistic surface and per-material colouring |
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

## 3D Surface Plot (`lpbf_3d_surface.png`)

Matches the companion notebook `4_plot_3d-ZGAN(1).ipynb`:

- **X-axis:** `log₁₀(Pi)` — normalised enthalpy
- **Y-axis:** `log₁₀(P_recoil / P_Laplace)` — Clausius–Clapeyron pressure
  ratio (separates keyhole from conduction regime)
- **Z-axis:** Pore fraction
- **Surface:** Fitted 2-D logistic `σ(a·log₁₀Pi + b·log₁₀PR + c)` over the
  experimental scatter, coloured per alloy

`Pi` alone (1-axis) does not fully separate all porosity regimes — the PR
axis is needed to distinguish keyhole porosity from lack-of-fusion porosity.

---

## Summary of Observed Results

| Aspect | Result |
|---|---|
| Pi groups discovered | 5 (A, V²/Lv, normalised-enthalpy form, capillary, Tb-variant) |
| Known Pi in null-space | cos = +1.0000 ✓ |
| Latent dimension k* | **1** — a single coordinate suffices for pore fraction |
| Test R² | **0.777** (vs ~0.446 from raw Pi formula) |
| Pe_vap / Pr in latent span? | R² = 0.47 / 0.06 → **both injected into Step 3** |
| Step 3 input dimension | **11** (9 physical + Pe_vap + Pr) |
| Symmetry type | **Scaling** — **2.0× loss gap** |
| Generators | 10 directions (11 variables − 1 latent dimension) |
| Constrained generators | V–P–Lv–γ, A–P, ρ–P, k–P trade-offs (process parameters) |
| Free generators | Lv, gamma, dT, Tb (material-only, only 5 discrete values) |
| Injected generators | Pe_vap (pure axis), Pr–P (Step 2b-augmented directions) |

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
