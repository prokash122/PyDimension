# Porous Media Flow — Local Symmetry Discovery in Two Ergun Regimes

This example runs the PyDimension Stage-1 symmetry-discovery pipeline
**separately** on the two limiting regimes of porous-media flow and shows
that the pipeline recovers the correct local scaling symmetry of each
regime from data alone — and, in each regime, that the actual Ergun
equation can be read directly off the winning scaling encoder's
L2-normed weight vector, with no post-processing beyond fitting an
overall scale α and a prefactor C.

The ground truth (never shown to the pipeline) is the Ergun equation

```
f = [ 150·(1−φ)/Re_p + 1.75 ] · (1−φ)/φ³
```

with friction factor `f = dP_L·d/(ρ·v²)`, particle Reynolds number
`Re_p = ρ·v·d/μ`, and porosity `φ`. It is a **sum of two power laws**,
so it has no single global scaling symmetry — but each limit does:

| Region | Dominant physics | Local law | Ergun exponents on (Re_p, φ, 1−φ) |
|---|---|---|---|
| **Viscous** (`Re_p/(1−φ) ≪ 85`) | Darcy drag | `f = 150 · Re_p⁻¹ · φ⁻³ · (1−φ)²` | `[−1, −3, +2]` |
| **Inertial** (`Re_p/(1−φ) ≫ 85`) | Forchheimer drag | `f = 1.75 · φ⁻³ · (1−φ)` | `[0, −3, +1]` — f independent of Re_p |

![Two regions on the Ergun master curve](ergun_two_regions.png)

---

## The Two Datasets

Both branches are generated directly from the textbook Ergun formula
with multiplicative log-normal noise, over a **wide porosity sweep** so
that `log φ` and `log(1−φ)` separate cleanly.

### Viscous branch — `dataset_ergun_viscous_widephi.csv`

`generate_viscous_dataset.py` samples the deep-Darcy limit
(`f = 150·Re_p⁻¹·φ⁻³·(1−φ)²`) with 5 % log-normal noise on f.

| Property | Value |
|---|---|
| Rows | 720 (30 Re_p × 12 φ × 2 d) |
| Re_p range | 9.3·10⁻⁷ … 1.0·10⁻³ — deep in the viscous branch |
| φ range | 0.15 – 0.85 — wide sweep, 12 values |
| Noise | 5 % log-normal on f, 3 % jitter on Re_p |

### Inertial branch — `dataset_ergun_inertial_widephi.csv`

`generate_inertial_dataset.py` samples the Forchheimer plateau
(`f = 1.75·φ⁻³·(1−φ)`) with the same noise model.

| Property | Value |
|---|---|
| Rows | 720 (30 Re_p × 12 φ × 2 d) |
| Re_p range | 9.3·10² … 1.0·10⁶ — viscous term below 5 % everywhere |
| φ range | 0.15 – 0.85 — wide sweep, 12 values |
| Noise | 5 % log-normal on f, 3 % jitter on Re_p |

**Schema (both CSVs):** `filename, tau, delta_p, dP_L, v, mu, rho, d,
phi, f, Re_p, f_ergun, steps, converged, stalled` — the physical
columns are pipeline inputs, `f` is the target, `Re_p`/`f_ergun` are
reference only.

---

## Variables — (1−φ) is included BEFORE dimensional analysis

The pipeline works with **6 input variables**: `rho, v, d, mu, phi,
one_minus_phi`, where `one_minus_phi = 1 − φ` is added as its own
variable before the Buckingham-Pi step. The Ergun porosity dependence
lives in both `φ` and `(1−φ)`, so with the solid fraction as an
explicit coordinate the scaling machinery can express the porosity
powers directly as global exponents (`[−1, −3, +2]` viscous,
`[0, −3, +1]` inertial).

`dP_L` is **excluded** from the Step-0 inputs because the target
`f = dP_L·d/(ρ·v²)` contains it linearly; keeping it in the basis
would put `f` into two of the four discovered Pi groups (as `f/Re_p`
and `f·Re_p`), making them redundant with the target. Dropping it
leaves a clean 3-Pi-group basis that maps one-to-one onto the Step-2
encoder inputs.

Dimension matrix (M, L, T × 6 variables, rank 3):

```
        rho    v    d    mu   phi  1-phi
Mass  [   1    0    0    1    0    0  ]
Len   [  -3    1    1   -1    0    0  ]
Time  [   0   -1    0   -1    0    0  ]
```

6 variables − rank 3 = **3 Pi groups**. The pipeline's null-space check
confirms `Re_p = ρvd/μ`, `φ`, and `(1−φ)` all lie in the discovered
span with cos = +1.0000 in each case.

**One caveat that matters for reading the results:** `φ` and `1−φ` can
never vary independently — every dataset lies on the 2-D manifold
`Π₄ = 1 − Π₃` inside the 3-D log-Pi space `(log Re_p, log φ, log(1−φ))`.
The component of the encoder direction *normal* to that manifold is
therefore unconstrained by any fit, so the raw 3-D cosine against the
Ergun exponents is not expected to reach ±1 in every run. The pipeline
reports both the raw 3-D cosine and the **manifold-projected cosine**
(using `dlog(1−φ) = −φ̄/(1−φ̄)·dlog φ`), which is the identifiable
quantity.

---

## Pipeline (per region)

`discover_symmetry.py` runs the same four Stage-1 steps on whichever
dataset it is given:

- **Step 0 — Buckingham-Pi reduction** via
  `pydimension.data_preprocessing.DataPreprocessor` (null-space +
  SymPy primitive-integer basis) from the explicit 3×6 dimension
  matrix above. Writes an augmented CSV carrying the derived
  `one_minus_phi` column to `_da_repo/`.
- **Step 1 — Normalisation.** Target `log10(f)` min-max scaled; Step-2
  features `[log10(Re_p), φ, 1−φ]` min-max scaled; Step-3 inputs are
  the Pi values `(Re_p, φ, 1−φ)` geometric-mean-centred (a purely
  multiplicative rescaling — no min-max — so the scaling encoder's
  internal `log` sees clean centred log-Pi coordinates).
- **Step 2 — Latent dimension.** MLP encoder `[3 → 64 → 32 → k]`,
  sweep `k = 1, 2, 3`. Optimal `k* = 1` in both regions
  (R²_test ≥ 0.995).
- **Step 3 — Symmetry type.** Competing scaling (`log|X|`),
  translational (`X`) and rotational (`X²`) single-linear encoders
  jointly trained with a shared Tanh-MLP decoder, plain L2
  (`weight_decay = 1e-4`). Scaling wins in both regions. The winning
  encoder's L2-normed weight vector is what the equation-discovery
  step below reads.

---

## Discovered vs Actual Equations — from the encoder's L2 weight vector only

Once Step 3 has declared `k* = 1` and `symmetry = scaling`, those two
facts alone certify that the friction factor takes the form of a
single monomial in the Pi inputs times a scalar function of that
monomial:

```
f  =  F ( Re_p^a · φ^b · (1−φ)^c )     with  (a, b, c) ∝ encoder weight w
```

To turn the L2-normed direction `w` into a *numerical* law:

1. Take the winning scaling encoder's row and **L2-normalise** it
   (`w = W / ‖W‖`), sign-aligned so `w · [log Re, log φ, log(1−φ)]`
   correlates positively with `log f`.
2. Fit the overall magnitude of the direction and the prefactor by
   plain 1-D OLS on
   `log f  =  α · ( w · [log Re, log φ, log(1−φ)] )  +  log C`.
3. Report the discovered law as `f = C · Re^(αw₁) · φ^(αw₂) · (1−φ)^(αw₃)`.

That is the *entire* extraction — no L1, no OLS on the three log
features separately, no snap-to-integer.

### Wide-φ synthetic **viscous** — `dataset_ergun_viscous_widephi.csv`

Standalone scaling encoder (`z = w · log|π|`) + Tanh MLP decoder
(`[1 → 64 → 64 → 1]`, Tanh) trained jointly for 1500 epochs, Adam
`lr = 1e-3`, `weight_decay = 1e-4`, 3 random restarts, best test-loss
kept (see `discover_equation_encoder_l2.py`).

Committed run (`--split-seed 42`, best restart seed = 2), full report
in `output_viscous_widephi/discovered_equation.txt`:

```
Best restart      : train MSE = 1.1e-5,   test MSE = 1.3e-5
Encoder w (L2)    : [ −0.2681, −0.7972, +0.5409 ]
cos vs Ergun      : +1.0000   (raw 3-D)      +1.0000  (manifold-projected)
α (1-D OLS)       : +3.7342
C (from log-OLS)  : 153.7
```

| | Equation | R²(f) |
|---|---|---|
| **Actual (Ergun deep Darcy)** | `f = 150.0 · Re_p⁻¹·⁰⁰⁰ · φ⁻³·⁰⁰⁰ · (1−φ)⁺²·⁰⁰⁰` | 1.000 |
| **Discovered (encoder L2 only)** | **`f = 153.7 · Re_p⁻¹·⁰⁰¹ · φ⁻²·⁹⁷⁷ · (1−φ)⁺²·⁰²⁰`** | **0.998** |

Prefactor within 3 %, all three exponents within 3 %.

### Wide-φ synthetic **inertial** — `dataset_ergun_inertial_widephi.csv`

Same architecture, same training recipe, same seed sweep.

Committed run (`--split-seed 42`, best restart seed = 1), full report
in `output_inertial_widephi/discovered_equation.txt`:

```
Best restart      : train MSE = 5.2e-5,   test MSE = 5.7e-5
Encoder w (L2)    : [ −0.0011, −0.9484, +0.3172 ]
cos vs Ergun      : +1.0000   (raw 3-D)      +1.0000  (manifold-projected)
α (1-D OLS)       : +3.1657
C (from log-OLS)  : 1.833
```

| | Equation | R²(f) |
|---|---|---|
| **Actual (Ergun Forchheimer plateau)** | `f = 1.750 · Re_p⁰ · φ⁻³·⁰⁰⁰ · (1−φ)⁺¹·⁰⁰⁰` | 1.000 |
| **Discovered (encoder L2 only)** | **`f = 1.833 · Re_p⁻⁰·⁰⁰⁴ · φ⁻³·⁰⁰² · (1−φ)⁺¹·⁰⁰⁴`** | **0.997** |

Prefactor within 5 %, `Re_p` exponent essentially zero (`−0.004`),
`φ` within 0.07 % of `−3`, `(1−φ)` within 0.4 % of `+1`.

### One-line summary

Both branches of the textbook Ergun equation — the viscous Darcy law
`f = 150·Re⁻¹·φ⁻³·(1−φ)²` and the inertial Forchheimer plateau
`f = 1.75·φ⁻³·(1−φ)` — are recovered end-to-end from just the winning
scaling encoder's L2-normed weight vector, with only a single 1-D OLS
fit to set the overall scale α and the prefactor C. The encoder alone
carries the physics; α and C are one line each.

### Identifiability caveat

Because `φ` and `1−φ` are algebraically linked on the data manifold,
the raw 3-D encoder direction is not uniquely determined by the loss —
there is a one-parameter equivalence class of `(w, F_decoder)` pairs
that all achieve the same MSE, and different random seeds/init can
land the joint optimiser in different members. On the seeds reported
above, the standalone runs converged to the physical member in both
regions; the identifiable *manifold-projected* direction is +1.0000 in
every restart, but the raw 3-D direction that turns into readable
exponents depends on which basin the optimiser settles in. The wide-φ
sweep makes the physical basin the deepest minimum and hence the
easiest to find, but does not make it the unique minimum.

---

## How to Run

```bash
cd projects/20260912_Stage1_Prokash/Examples/porous_media_lbm_symmetry

# Generate the two wide-φ synthetic datasets (only needed once)
python generate_viscous_dataset.py   --phi-min 0.15 --phi-max 0.85 \
    --n-phi 12 --output dataset_ergun_viscous_widephi.csv
python generate_inertial_dataset.py  --phi-min 0.15 --phi-max 0.85 \
    --n-phi 12 --output dataset_ergun_inertial_widephi.csv

# Step 1: run the Stage-1 pipeline on each region (produces run.log,
# lbm_*.png diagnostic figures, and _da_repo/ dimensional-analysis
# artifacts).
python discover_symmetry.py --data dataset_ergun_viscous_widephi.csv \
    --output-dir output_viscous_widephi --seed 42 \
    --latent-epochs 300 --sym-epochs 600 --n-restarts 3
python discover_symmetry.py --data dataset_ergun_inertial_widephi.csv \
    --output-dir output_inertial_widephi --seed 42 \
    --latent-epochs 300 --sym-epochs 600 --n-restarts 3

# Step 2: extract the numerical law from the winning scaling encoder's
# L2-normed weight vector.  Writes discovered_equation.txt and
# trained_encoder_l2.pt into the output directory.
python discover_equation_encoder_l2.py \
    --data dataset_ergun_viscous_widephi.csv \
    --out output_viscous_widephi --region viscous --split-seed 42
python discover_equation_encoder_l2.py \
    --data dataset_ergun_inertial_widephi.csv \
    --out output_inertial_widephi --region inertial --split-seed 42
```

`discover_symmetry.py` tees its console transcript to `run.log` and
saves diagnostic plots. `discover_equation_encoder_l2.py` is a small
standalone script that re-trains just the scaling encoder + Tanh MLP
decoder pair (matches the pipeline's Step-3 architecture), reads the
L2-normed encoder direction, fits `α` and `C` by 1-D OLS on
`log f = α·(w·x) + log C`, and writes the discovered law to
`discovered_equation.txt`. The trained encoder + decoder state dict
is saved to `trained_encoder_l2.pt` for reproducibility.

`run_two_regions.py` bundles both `discover_symmetry.py` runs and
locks `OMP/MKL/OPENBLAS_NUM_THREADS=1` and `PYTHONHASHSEED=0` for
bit-reproducibility of the committed logs.

---

## Output Files

Per region (`output_viscous_widephi/`, `output_inertial_widephi/`):

| File | Contents |
|---|---|
| `run.log` | Full console transcript of the Stage-1 pipeline |
| `discovered_equation.txt` | Encoder-L2 equation-extraction report (best-restart w, cos vs Ergun, α, C, discovered law, R²) |
| `trained_encoder_l2.pt` | Saved encoder + decoder state dict for the best restart, plus `w_raw`, `w_norm`, `alpha`, `logC` |
| `lbm_ergun_collapse.png` | `f·φ³/(1−φ)` vs `Re_p/(1−φ)` — collapse onto the textbook curve |
| `lbm_pi_candidates.png` | Pi-basis heatmap + `log f` vs each `log Πₖ` |
| `lbm_symmetry_discovery.png` | Symmetry-type bar chart + latent-dim R² curve |
| `_da_repo/dimension_matrix.csv` | 3×6 integer dimension matrix fed to `DataPreprocessor` |
| `_da_repo/dataset_with_one_minus_phi.csv` | Augmented copy of the dataset with the derived `one_minus_phi` column |
| `_da_repo/data/basis_vectors.csv` | Primitive integer Pi-exponent vectors (`Re_p`, `φ`, `1−φ`) |
| `_da_repo/data/afterDA_data.csv` | Normalised Pi values per row |

Top level: `ergun_two_regions.png` — both datasets on the Ergun master
curve.

---

## File Organisation

```
porous_media_lbm_symmetry/
├── README.md                                 ← this file
├── dataset_ergun_viscous_widephi.csv         ← 720 synthetic rows (viscous branch)
├── dataset_ergun_inertial_widephi.csv        ← 720 synthetic rows (inertial branch)
├── generate_viscous_dataset.py               ← wide-φ viscous generator
├── generate_inertial_dataset.py              ← wide-φ inertial generator
├── plot_two_regions.py                       ← master-curve overview figure
├── discover_symmetry.py                      ← Stage-1 pipeline (region-agnostic)
├── discover_equation_encoder_l2.py           ← reads L2-normed encoder weight → law
├── run_two_regions.py                        ← one-command runner for both regions
├── ergun_two_regions.png
├── output_viscous_widephi/
└── output_inertial_widephi/
```

Dependencies: `torch`, `numpy`, `pandas`, `scipy`, `sympy`,
`matplotlib`, `seaborn`. Install with `pip install -r
requirements.txt` from the repo root.
