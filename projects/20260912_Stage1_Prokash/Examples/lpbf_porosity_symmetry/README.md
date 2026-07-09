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

## DA-Only Ablation (`--da-only`)

Running the pipeline on **only the 6 DA-discovered Pi groups** — without
appending the known `Pe_vap` and `Pr` columns — removes the collinearity
caveat entirely (`log(Pe_vap)` lies exactly in the DA span, so the
8-feature input has rank ≤ 7 and gauge directions in its null space; the
6-feature input has none):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python discover_symmetry.py --data dataset_lpbf.csv --seed 42 --da-only \
  --output-dir output_lpbf_da_only
```

`Pe_vap` and `Pr` are still computed: Step 2b keeps its diagnostic role,
and `Pe_vap` appears in reports/figures through its **exact coordinate
representation in the DA log-Pi basis** (the lstsq solution of
`pi_basis · c = Pe_vap-exponents`, which is exact since cos = +1.0000).
`Pr` has no exact DA representation (`η`, `Cp` are not pipeline
variables), so it is omitted from the DA-only figures.

**Observed results (locked-environment runs, seeds 42 / 0 / 1):**

| Aspect | DA-only (6 Pi) | Default (8 Pi incl. Pe_vap, Pr) |
|---|---|---|
| k* (seed 42) | **1** — test R² 0.755 (k=2: 0.738) | 2 — test R² 0.768 (k=1: 0.747) |
| k* across seeds | 1 / 1 / 3 | 2 (seed 42) |
| Step 3 winner, seed 42 | **scaling** (1.2×) | scaling (1.3×) |
| Step 3 winner, seed 0 | **scaling** (1.1×) | translational (~1.0×) |
| Step 3 winner, seed 1 | **scaling** (1.1×) | translational (~1.1×) |
| Rotational | always last or clearly beaten | always excluded (2–4×) |
| Generators | 5 (6 − 1), no gauge directions | 6 (8 − 2), some gauge |
| cos(W, Pe_vap ref) | +0.07 / −0.31 / −0.88 by seed | weak (rows seed-dependent) |

Two observations worth noting:

1. **The scaling-vs-translational tie resolves in favour of scaling.**
   With the redundant known-Pi columns removed, *scaling wins on all three
   hash-locked seeds* (margins 1.1–1.2×), whereas the 8-feature input
   flips winner across seeds. The margins remain modest, but the
   direction is now consistent.
2. **A single discovered coordinate collapses the data better than the
   hand-derived group.** A logistic fit of pore fraction against the
   DA-only 1-D latent `z = W · log(Pi_centred)` gives **R² = 0.75**,
   versus 0.46 for `log10(Pe_vap)` on the same 232 points (and ≤ 0.55
   for every individual DA Pi candidate). The discovered direction is
   *not* aligned with Pe_vap (cos = +0.07 at seed 42; sample-wise
   Pearson corr(z, log₁₀Pe_vap) = +0.60) — the pipeline finds a
   different, better-collapsing invariant coordinate rather than
   rediscovering the textbook one. The W direction itself remains
   seed-dependent; the collapse quality is the stable statement.

Outputs are committed under `output_lpbf_da_only/` (figures + `run.log`
from the locked seed-42 run above).

---

## Known-Basis Mode (`--known-basis`) — best of both

The default 8-feature run keeps the known groups but is rank-deficient;
`--da-only` is full-rank but drops them. `--known-basis` keeps Pe_vap and
Pr as encoder features **and** restores full rank by a change of Pi basis
(always legitimate under Buckingham): the DA coordinate basis is rotated
so the Pe_vap direction (exactly in the DA span, cos = +1) becomes one
explicit basis vector, and the 5 orthogonal-complement combinations
`Pi⊥1…Pi⊥5` replace the 6 DA groups. With Pr (genuinely outside the DA
span — `η`, `Cp` are not pipeline variables) this gives a **full-rank
7-feature set** with the known groups as literal feature axes and no
gauge directions in the generator null space.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python discover_symmetry.py --data dataset_lpbf.csv --seed 42 --known-basis \
  --output-dir output_lpbf_known_basis
```

**Observed results (locked-environment runs, seeds 42 / 0 / 1):**

| Aspect | Known-basis (7 feats) | DA-only (6) | Default (8) |
|---|---|---|---|
| Step 3 winner, seed 42 | **scaling, 1.4×** | scaling, 1.2× | scaling, 1.3× |
| Step 3 winner, seed 0 | **scaling, 1.6×** | scaling, 1.1× | translational, ~1.0× |
| Step 3 winner, seed 1 | **scaling, 1.3×** | scaling, 1.1× | translational, ~1.1× |
| Best test R² (seed 42/0/1) | 0.772 / **0.848** / **0.873** | 0.755 / 0.816 / 0.838 | 0.768 / – / – |
| k* across seeds | 4 / 1 / 2 | 1 / 1 / 3 | 2 (seed 42) |
| Rank of Step 3 input | full (7) | full (6) | ≤ 7 of 8 (gauge) |
| Pe_vap linearly in z (Step 2b, seed 42) | **R² = 0.87 ✓** | 0.42 | 0.77 |
| Max Pe_vap drift along generators (seed 42) | **0.21** | 0.90 | 0.49 |

Takeaways:

1. **Scaling wins on every seed with the largest margins of any
   configuration** (1.3–1.6× vs translational/rotational). Removing the
   collinearity while keeping the known groups sharpens the symmetry-type
   competition rather than weakening it.
2. **Prediction improves too**: best test R² rises to 0.85–0.87 on seeds
   0/1 (vs 0.77 for the default 8-feature run) — the encoder no longer
   spends capacity disentangling duplicated directions.
3. **The known physics is actually absorbed**: at seed 42 the latent `z`
   linearly encodes log₁₀(Pe_vap) with R² = 0.87 (the only configuration
   to clear the 0.80 "discovered" threshold), and the generator orbits
   move Pe_vap by at most 21% at |ε| = 0.5 (vs 49% default, 90% DA-only)
   — the discovered null space is closest to the known invariance here.
4. Remaining honest caveat: **k\* is still seed-dependent** (4/1/2) with
   a flat R²(k) curve — the latent dimension is not sharply identified on
   232 noisy points; report it as "1–2 effective coordinates" with
   multi-seed statistics rather than a single k*.

Outputs are committed under `output_lpbf_known_basis/` (figures +
`run.log` from the locked seed-42 run above).

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

# Ablation: only the 6 DA Pi groups as encoder features (no Pe_vap/Pr columns)
python discover_symmetry.py --data dataset_lpbf.csv --da-only

# Known-basis: keep Pe_vap/Pr but rotate the DA basis so Pe_vap is an explicit
# axis — full-rank 7-feature set (5 complement groups + Pe_vap + Pr)
python discover_symmetry.py --data dataset_lpbf.csv --known-basis

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
