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
Pi = (Lv · rho · A · P · V) / (k² · (Tb − Tm). (Tm-T0)
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
literature are computed per row and appended to the Pi feature set as
**known extra Pi groups** — they are first-class encoder inputs for both
Step 2 and Step 3 (Step 2b remains as a diagnostic of whether the latent
`z` actually absorbed them):

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
   `W/(m·K)`). This is the only Pi-basis path the script supports — the
   import is hard-required, so a missing `seaborn` raises rather than
   silently swapping in a fallback.

**Actual output — 6 Pi groups discovered:**

```
π1 = A                                     (absorptivity is dimensionless → own group)
π2 = V² × Lv⁻¹                             (kinetic / latent heat ratio)
π3 = P × V³ × rho × k⁻² × dT⁻²            (normalised enthalpy form)
π4 = P × V × rho × gamma⁻²                 (capillary group)
π5 = P × V³ × rho × k⁻² × Tb⁻²            (normalised enthalpy with Tb)
π6 = P × V³ × rho × k⁻² × Tm_minus_T0⁻²   (normalised enthalpy with Tm − T0)
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
- `X_norm_step2` — 8 Pi features (6 DA groups + Pe_vap + Pr) scaled to
  [0, 1] (input to Step 2).
- `pi_centred` — the same 8 Pi **values** divided by their per-column
  geometric mean (input to Step 3; purely multiplicative, no min-max).
- `y_norm` — min-max scaled pore fraction.

---

### Step 2 — Latent Dimension Discovery

**Goal:** Find the smallest `k*` that can predict pore fraction.

**Input:** The **8 Pi features** — 6 DA groups plus the known `Pe_vap`
and `Pr` (pi-only mode, default).

> Pi-only is the recommended default for this dataset. The LPBF material-property
> columns (`A`, `rho`, `k`, `Lv`, `dT`) only take 5 discrete values (one per
> alloy) — after min-max scaling, entire material groups collapse to 0, and
> `log(0)` would produce NaN. The Pi features are always well-defined.

**Encoder architecture:** Multilayer MLP (default hidden dims `[64, 32]`):
```
8  →  Linear(64)  →  ReLU  →  Linear(32)  →  ReLU  →  Linear(k)
```
Paired with a nonlinear decoder (two 64-unit hidden layers). Sweep over
`k = 1, 2, 3, 4`.

**Actual results (with Pe_vap and Pr appended):**

| k | R2_train | R2_test | MSE |
|---|---|---|---|
| 1 | 0.8920 | 0.7472 | 0.025400 |
| **2** | **0.8902** | **0.7680** | **0.023316** ← optimal |
| 3 | 0.9093 | 0.7611 | 0.024006 |
| 4 | 0.9067 | 0.7639 | 0.023726 |

**`k* = 2`** — with the two known groups added the best test R² moves to
two latent coordinates (0.768 vs 0.747 at k = 1). The margin is small:
this experimental dataset is noisy and pore fraction saturates at 0 and 1,
so a single coordinate already captures most of the signal.

---

### Step 2b — Does `z` Encode Pe_vap and Pr?

**Goal:** Diagnostic only (Pe_vap and Pr are now direct encoder inputs, so
nothing is injected anywhere): check whether the trained latent coordinates
actually absorbed the two known groups.

**What it does:**
1. Evaluates the trained Step 2 encoder on every sample → `z` of shape
   `(232, k*)` (here `k* = 2`).
2. Fits two ordinary linear regressions, `z → log10(Pe_vap)` and
   `z → log10(Pr)`, and reports the R² of each.

**Actual results (Pe_vap and Pr as encoder inputs):**

| Quantity | R²(`z` → log10·) | Verdict |
|---|---|---|
| Pe_vap | 0.7692 | just below the 0.80 threshold — largely absorbed |
| Pr     | 0.4650 | partially absorbed |

Even as direct inputs, neither is *linearly* recoverable from the 2-D
latent with R² ≥ 0.8 — the encoder mixes them nonlinearly with the DA
groups rather than passing either through unchanged.

---

### Step 3 — Symmetry Type Identification

**Goal:** Determine whether the invariance is scaling, translational, or
rotational.

**Encoder input:** the 8 Pi **values** (6 DA groups + Pe_vap + Pr), each
column divided by its geometric mean — a purely multiplicative rescaling.

> Pi-only Step 3: because the centring is multiplicative, the scaling
> encoder's internal `log` sees centred log-Pi coordinates — no
> `log(log(·))` degeneracy — and the discovered weight vector and
> generators live directly in dimensionless Pi space. This also sidesteps
> the 5-alloy min-max collapse problem that made raw-X Step 3 fragile.

**What it does:** Trains three competing single-linear-layer encoders:

| Encoder | Transform | What it tests |
|---|---|---|
| Scaling | `z = W · log(Pi)` | Power-law / dimensional symmetry |
| Translational | `z = W · Pi` | Additive / affine symmetry |
| Rotational | `z = W · Pi²` | Quadratic / Euclidean symmetry |

**Actual results (k* = 2, 8 Pi inputs; reproducible run, seed 42 with
`PYTHONHASHSEED=0` and single-threaded BLAS):**

```
scaling        : 0.016873  ← winner
translational  : 0.022566
rotational     : 0.075993
Loss gap: 1.3×
```

**Caveat — scaling vs translational is a statistical tie on this
dataset.** Across hash-locked seeds the winner flips (seed 42: scaling by
1.3×; seeds 0 and 1: translational by ~1.0–1.1×), while rotational always
loses by 2–4×. This is expected: after geometric centring the Pi values
sit near 1, where `log(Pi) ≈ Pi − 1`, so the scaling and translational
encoders see nearly identical features on a noisy dataset whose output
saturates at 0 and 1. The robust conclusions are (a) rotational symmetry
is excluded and (b) the 2-D latent organises pore fraction cleanly — the
scaling-vs-affine distinction is below this dataset's resolution.

---

### Step 4 — Generator Extraction

**Goal:** Extract directions in log-Pi-space along which pore fraction
is invariant.

**Input:** Winning encoder weight matrix `W` (shape `2 × 8` — the 6 DA
Pi groups plus Pe_vap and Pr; `k* = 2`).

**Encoder weight rows (L2-normalised, reproducible seed-42 run):**

```
          Pi1      Pi2      Pi3      Pi4      Pi5      Pi6   Pe_vap       Pr
Row 1: -0.099   -0.685   -0.472   +0.223   -0.331   +0.319   +0.192   +0.008
Row 2: -0.056   -0.141   +0.239   +0.115   +0.467   -0.285   +0.324   -0.709
```

Direction cosines against the two natural references:
`cos(row1, known-Pi DA coords) = +0.07`, `cos(row2, ·) = +0.19`;
`cos(row1, pure Pe_vap axis) = +0.19`, `cos(row2, ·) = +0.32`.
The alignment is weak — the encoder spreads the signal across the
redundant Pi set instead of isolating the textbook direction. See the
collinearity caveat below, and note the row values themselves are
seed-dependent (the 2-D *column space* of `W` is the meaningful object).

With `k* = 2` and an 8-D Pi input there are `8 − 2 = 6` null-space
generators.

---

### Step 5 — Physical Interpretation

**Actual generators (8 Pi inputs, 2 latent → 6 generators):** each is a
direction in log-Pi space; e.g. generator 5 increases `Pe_vap` while
adjusting the DA groups to keep pore fraction fixed, and generator 2 is
dominated by the capillary group `π4` (surface tension barely affects
pore fraction in this dataset).

**Caveat — the extended Pi set is collinear.** `Pe_vap` is itself
dimensionless, so `log(Pe_vap)` lies **exactly** in the span of the six
DA log-Pi columns (`Pr` is close to their span too, since it takes only
5 per-alloy values). The 8-column Step 3 input therefore has rank ≤ 7,
and the null space of `W` contains directions that exchange `Pe_vap`
against the equivalent DA-group combination — a gauge freedom of the
redundant parameterisation, not new physics. Individual generator
components should be read with that in mind; the invariant statement is
the 2-D column space of `W`, not any particular null-space basis.

**5-alloy confounding still applies.** All material-property groups take
at most 5 distinct values (one per alloy), so generators dominated by
them are data-limited. Only `P`- and `V`-driven variation (entering
π2–π6 and Pe_vap) is continuous within a material.

---

## Output Files

All outputs go to `output_lpbf_porosity_symmetry/` (configurable via
`--output-dir`).

| File | Contents |
|---|---|
| `lpbf_pi_candidates.png` | Pi-basis exponent heatmap + scatter of pore fraction vs each `log₁₀(Πₖ)` with logistic fit and R² |
| `lpbf_porosity_symmetry_discovery.png` | 3-panel: Pi-collapse, symmetry-type bar chart, discovered iso-invariant orbits in (log V, log P) |
| `lpbf_discovered_law_generators.png` | 4-panel (Pi space): coefficient heatmap of the two `W` rows vs the known-Pi DA coordinates and pure `Pe_vap` axis, pore fraction over the 2-D discovered latent `(z₁, z₂)`, heatmap of the 6 null-space generators, and orbit-invariance check (latent `z` exactly flat; known Pi drifts, reflecting W–Pi misalignment and gauge directions) |
| `_da_repo/dataset_lpbf_enriched.csv` | CSV enriched with `gamma`, `Tb` columns |
| `_da_repo/dimension_matrix.csv` | Explicit dimension matrix fed to `DataPreprocessor` |
| `_da_repo/basis_vectors.csv` | Integer Pi-group exponent vectors |

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/lpbf_porosity_symmetry

# Reproducible run (matches committed run.log — locked hash seed + threads)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python discover_symmetry.py --data dataset_lpbf.csv --seed 42

# Default: Pi-only input to Steps 2 and 3 (recommended); Step 3 winner is
# seed-sensitive on this dataset without the locked environment above
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
| Pi groups discovered | 6 (A, V²/Lv, normalised-enthalpy form, capillary, Tb- and (Tm−T0)-variants) |
| Known Pi in null-space | cos = +1.0000 ✓ |
| Extra known Pi used | **Pe_vap and Pr appended as Step 2/3 encoder inputs** (8 Pi features) |
| Latent dimension k* | **2** — test R² 0.768 (vs 0.747 at k = 1; small margin) |
| Pe_vap / Pr linearly in latent span? | R² = 0.77 / 0.47 — absorbed nonlinearly, not passed through |
| Step 3 encoder input | 8 geometric-mean-centred Pi values |
| Symmetry type | **Scaling** (seed 42, 1.3×) — but a statistical tie with translational across seeds; rotational always excluded (2–4×) |
| Generators | 6 directions in log-Pi space (8 Pi inputs − 2 latent dims) |
| Caveat | log(Pe_vap) is exactly in the DA-group span → rank ≤ 7, some null-space directions are gauge |

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
