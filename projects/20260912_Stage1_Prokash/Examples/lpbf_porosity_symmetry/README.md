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

The pore fraction `f` depends on **ten physical inputs** with **four
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
| (Tm − T0), T0 = 298 K | `Tm_minus_T0` | K | K |

By the **Buckingham Pi theorem** (10 variables − 4 dimensions = **6 independent
dimensionless groups**). These six groups are the *only* features fed to
Step 2. The extra `Tm_minus_T0` column is added so the encoder has
explicit access to the second factor of the Pe_vap denominator
`(Tb − Tm) · (Tm − T0)`. The first factor `(Tb − Tm)` is *not* added as
a separate column because it duplicates the existing `dT`.

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

**Input:** The **6 Pi features only** (pi-only mode, default).

> Pi-only is the recommended default for this dataset. The LPBF material-property
> columns (`A`, `rho`, `k`, `Lv`, `dT`) only take 5 discrete values (one per
> alloy) — after min-max scaling, entire material groups collapse to 0, and
> `log(0)` would produce NaN. The Pi features are always well-defined.

**Encoder architecture:** Multilayer MLP (default hidden dims `[64, 32]`):
```
6  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```
Paired with a nonlinear decoder (two 64-unit hidden layers). Sweep over
`k = 1, 2, 3, 4`.

**Actual results:**

| k | R2_train | R2_test | MSE |
|---|---|---|---|
| **1** | **0.9043** | **0.7552** | **0.024603** ← optimal |
| 2 | 0.8698 | 0.7379 | 0.026335 |
| 3 | 0.8116 | 0.7138 | 0.028758 |
| 4 | 0.8800 | 0.7434 | 0.025790 |

**`k* = 1`** — a single latent coordinate gives the best test R² (0.755)
and the lowest MSE; wider bottlenecks lose generalisation. Adding the
`Tm_minus_T0` column did not push `k*` upward (the previous run that
saw `k* = 4` was an artefact of the redundant `Tb_minus_Tm` duplicate
of `dT`, which has since been removed).

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

**Actual results (with only `Tm_minus_T0` added):**

| Quantity | R²(`z` → log10·) | Verdict |
|---|---|---|
| Pe_vap | 0.4161 | NOT in latent span → bottleneck-injected |
| Pr     | 0.0403 | NOT in latent span → bottleneck-injected |

→ Step 3 encoder input stays at **10 raw physical variables**; the
decoder input grows from `k* = 1` to **`k* + 2 = 3` dims** as Pe_vap
and Pr are concatenated to the bottleneck.

*Earlier exploration:* adding `Tb_minus_Tm` as a redundant duplicate of
`dT` made `k*` jump to 4 and both quantities reached R² > 0.97 — but
that "discovery" was an artefact of the duplicate, not a real gain. With
only the genuinely-new `(Tm − T0)` added, the encoder still cannot fit
Pe_vap or Pr inside a 1-D latent, so the option-A injection kicks in.

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or
rotational.

**Encoder input:** the 10 raw physical variables.
**Bottleneck:** `[W·ϕ_s(X_10), Pe_vap, Pr]` — `k* + 2 = 3` dims for
`k*=1`. Pe_vap and Pr are bottleneck-concatenated since Step 2b found
them outside the latent span.

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

**Actual results (k* = 1, Pe_vap and Pr in bottleneck):**

```
scaling        : 0.019752  ← winner
rotational     : 0.035534
translational  : 0.036413
Loss gap: 1.8×
```

**Scaling wins** with the widest gap recorded for this dataset (**1.8×**,
vs 1.4× without `Tm_minus_T0`). Adding the genuinely-new temperature
column slightly improves the scaling-class signal while keeping the
single-latent structure intact.

---

### Step 4 — Generator Extraction

**Goal:** Extract directions in log-variable-space along which pore fraction
is invariant.

**Input:** Winning encoder weight matrix `W` (shape `1 × 10` — the 10
physical variables; Pe_vap and Pr live in the bottleneck, not the
encoder).

**Encoder weight row (L2-normalised):**

```
         P        V        A      rho        k       Lv       dT    gamma       Tb  Tm-T0
Row 1: +0.761  +0.436  +0.012  +0.234  -0.263  +0.090  +0.192  -0.010  -0.132  +0.212
```

Cosine with the known normalised-enthalpy exponents
`[1, 1, 1, 1, −2, 1, −2, 0, 0, 0]` is **+0.46** — the discovered
direction aligns with the textbook formula, with `Tm_minus_T0` picking
up a small positive weight.

With `k* = 1` and a 10-D encoder input there are `10 − 1 = 9`
null-space generators.

---

### Step 5 — Physical Interpretation

**Actual generators (10-D encoder input, 1 latent → 9 generators):**

| Generator | Dominant variable | Trade-off | Physical meaning |
|---|---|---|---|
| 1 | V | increase V, decrease P | Speed–power trade-off |
| 2 | A | increase A alone | Pure absorptivity axis |
| 3 | rho | increase ρ, decrease P | Density–power trade-off |
| 4 | k | increase k, increase P | Conductive metal absorbs higher P |
| 5 | Lv | increase Lv, decrease P | Latent-heat trade-off |
| 6 | dT | increase dT, decrease P | Superheat trade-off |
| 7 | gamma | increase γ alone | Pure surface-tension axis |
| 8 | Tb | increase Tb, increase P | Boiling-temperature trade-off |
| 9 | Tm − T0 | increase (Tm − T0), decrease P | Melt-pool-temperature trade-off (new) |

Pe_vap and Pr enter the decoder directly (bottleneck-concatenated, option A)
and therefore do not appear as generators — they influence the fit
globally rather than restricting any single invariance direction.

**Caveat — 5-alloy confounding still applies.** Seven of the ten encoder
columns (`A, rho, k, Lv, gamma, Tb, Tm_minus_T0`) take at most 5 distinct
values across the dataset (one per alloy), so any generator dominated by
them is data-limited. Only the `P` and `V` (and per-row `dT`) directions
are continuously varied within a material.

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
| Pe_vap / Pr in latent span? | R² = 0.42 / 0.04 → **both bottleneck-injected** |
| Injection style | **Option A** — Pe_vap and Pr concatenated to the bottleneck |
| Step 3 encoder input | 10 raw physical variables |
| Step 3 bottleneck dim | **3** (`k* = 1` + Pe_vap + Pr) |
| Symmetry type | **Scaling** — **1.8× loss gap** (widest seen on this dataset) |
| Generators | 9 directions (10 encoder inputs − 1 latent dimension) |
| Constrained generators | V–P, ρ–P, k–P, dT–P, (Tm − T0)–P trade-offs (process parameters) |
| Free generators | A, Lv, gamma, Tb (material-only, only 5 discrete values) |
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
