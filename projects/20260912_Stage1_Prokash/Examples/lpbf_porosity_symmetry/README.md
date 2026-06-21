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
Step 2.

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
   **not in the latent span** and injected into Step 3 — but **into the
   bottleneck, not the encoder input** (Option A).

   The Step 3 encoder still acts only on the 9 raw physical variables,
   producing a `k*`-dim projection `W·ϕ_s(X_9)`. Pe_vap and Pr are min-max
   scaled and **concatenated to the encoder output** before the decoder
   sees them. The decoder therefore receives `[W·ϕ_s(X_9), Pe_vap, Pr]`,
   so the Pi quantities are given to it for free — the symmetry-class
   encoder only has to explain whatever is left after Pe_vap and Pr are
   already accounted for.

**Actual results:**

| Quantity | R²(`z` → log10·) | Verdict |
|---|---|---|
| Pe_vap | 0.4651 | NOT in latent span → bottleneck-injected |
| Pr     | 0.0592 | NOT in latent span → bottleneck-injected |

→ Step 3 encoder input stays at **9 features**; decoder input grows from
`k* = 1` to **`k* + 2 = 3` dims** (the extra two are Pe_vap and Pr).

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or
rotational.

**Encoder input:** the 9 raw physical variables (unchanged).
**Bottleneck:** `[W·ϕ_s(X_9), Pe_vap, Pr]` — `k* + 2 = 3` dims for `k*=1`.
The decoder reads this concatenation and only the `W·ϕ_s(·)` part is
subject to the per-class transform.

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

**Actual results (Pe_vap and Pr concatenated to the bottleneck):**

```
scaling        : 0.024219  ← winner
translational  : 0.032858
rotational     : 0.034098
Loss gap: 1.4×
```

**Scaling wins**, confirming the power-law dimensional structure of LPBF
porosity. With Pe_vap and Pr handed to the decoder for free, every class
has less to do — and the encoder no longer gets to weight them inside the
per-class transform. All three MSEs rise relative to the earlier
encoder-input runs, and the gap narrows further to **1.4×**. The
ranking still puts scaling first.

---

### Step 4 — Generator Extraction

**Goal:** Extract directions in log-variable-space along which pore fraction
is invariant.

**Input:** Winning encoder weight matrix `W` (shape `1 × 9` — only the 9
physical variables; Pe_vap and Pr live in the bottleneck, not the
encoder).

**Encoder weight row (L2-normalised):**

```
         P        V        A      rho        k       Lv       dT    gamma       Tb
Row 1: +0.561  +0.330  +0.058  +0.205  +0.105  -0.321  -0.575  +0.169  -0.239
```

Cosine with the known normalised-enthalpy exponents
`[1, 1, 1, 1, −2, 1, −2, 0, 0]` is **+0.49** — the discovered direction
aligns with the textbook formula (positive cos this time; the column-mix
chosen by the encoder, since Pe_vap and Pr no longer compete with it as
encoder inputs).

With `k* = 1` and a 9-D encoder input there are `9 − 1 = 8` null-space
generators in the encoder's null space (Pe_vap and Pr don't enter the
null-space calculation because they bypass the encoder).

---

### Step 5 — Physical Interpretation

**Actual generators (9-D encoder input, 1 latent → 8 generators):**

| Generator | Dominant variable | Trade-off | Physical meaning |
|---|---|---|---|
| 1 | V | increase V, decrease P | Speed–power trade-off |
| 2 | A | increase A, decrease P | Absorptivity–power compensation |
| 3 | rho | increase ρ, decrease P | Density–power trade-off |
| 4 | k | increase k, decrease P | Conductive metal absorbs lower P |
| 5 | Lv | increase Lv, increase P | Latent-heat trade-off |
| 6 | dT | increase dT, increase P | Superheat trade-off |
| 7 | gamma | increase γ, decrease P | Surface-tension trade-off |
| 8 | Tb | increase Tb, increase P | Boiling-temperature trade-off |

Pe_vap and Pr no longer appear as generators because they bypass the
encoder — the null-space calculation only involves the 9 physical
variables that the encoder actually weights.

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

- **Bottleneck-only (no generator):** Pe_vap and Pr enter the decoder
  directly, so they are not in the encoder's null space and there is no
  generator that names them. Their influence on the fit is global (they
  are given to the decoder for every sample) rather than restricted to
  a particular invariance direction.

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
| Pe_vap / Pr in latent span? | R² = 0.47 / 0.06 → **both injected into Step 3** |
| Injection style | **Option A** — Pe_vap and Pr concatenated to the bottleneck (skip encoder) |
| Step 3 encoder input | 9 raw physical variables |
| Step 3 bottleneck dim | **3** (`k* = 1` + Pe_vap + Pr) |
| Symmetry type | **Scaling** — **1.4× loss gap** |
| Generators | 8 directions (9 encoder inputs − 1 latent dimension) |
| Constrained generators | V–P, A–P, ρ–P, k–P trade-offs (process parameters) |
| Free generators | Lv, gamma, dT, Tb (material-only, only 5 discrete values) |
| Bottleneck-only quantities | Pe_vap, Pr (no generator — bypass the encoder) |

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
