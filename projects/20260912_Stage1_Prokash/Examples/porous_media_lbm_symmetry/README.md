# Porous Media LBM — Symmetry Discovery

Discover the hidden scaling symmetry in 3D porous media flow data generated
by a Lattice Boltzmann Method (LBM) simulator using the PyDimension Stage1
pipeline. The pipeline recovers the **Ergun-equation structure**
(`f = g(Re_p, φ)`) entirely from the LBM simulation data — without being
given any formula in advance.

> See **PLAN.md** for the full design rationale and expected pipeline behavior.

---

## Problem Statement

**Output (dimensionless):**
```
f = (dP_L · d) / (rho · v²)            friction factor
```

**Inputs (6 physical variables):**
```
dP_L, v, mu, rho, d, phi
```

**Correlation we are fitting:**
```
f = g(Re_p, phi)
```
where `Re_p = rho · v · d / mu` is the particle Reynolds number.

The ground-truth Ergun formula is `f = [150·(1−φ)/Re_p + 1.75] · (1−φ)/φ³`,
but the pipeline does not know this — it should recover the structure
(2 latent coords, scaling symmetry, 4 generators) from data alone.

---

## Dataset — `dataset_lbm_porous.csv`

| Property | Value |
|---|---|
| Rows | 144 converged LBM runs |
| Sphere radii | 2 (r = 3 and r = 5 LU; d = 6 and d = 10) |
| Porosities | 6 (phi ∈ {0.459, 0.502, 0.543, 0.551, 0.603, 0.609}) |
| Relaxation time `tau` | 0.9, 1.0, 1.1 (3 viscosities) |
| Driving force `delta_p` | 8 values: 1e-6 to 3e-4 |
| Re_p range | [3e-7, 1e-3] — **deep Darcy regime** |
| f range | [7.6e4, 4.6e8] — spans ~5 orders of magnitude |

**Schema:**

| Column | Description |
|---|---|
| `filename`, `tau`, `delta_p` | LBM run identifiers |
| `dP_L`, `v`, `mu`, `rho`, `d`, `phi` | 6 physical inputs (NN inputs) |
| `f` | **dimensionless friction factor — NN target** |
| `Re_p`, `f_ergun` | reference values (not NN inputs) |
| `steps`, `converged`, `stalled` | convergence diagnostics |

### Known limitations of this dataset

- **Only two particle sizes (d ∈ {6, 10})** — the d-direction generator
  is now constrained (it was undetermined in the earlier single-radius
  dataset) but a third radius would still help.
- **Deep Darcy regime only** (Re_p < 1e-3) — the inertial Forchheimer term
  (1.75) is invisible. The data follows pure `f ∝ 1/Re_p`.
- **f / f_ergun ≈ 0.66** — LBM gives ~34% lower drag than the textbook
  Ergun constant (150). This is fine for symmetry discovery (the functional
  form, not the prefactor, is what matters).

To address remaining limitations, the dataset can be extended with one
more sphere radius (e.g. `r = 4` or `r = 7`) and higher driving force
`delta_p ∈ {1e-3, 1e-2}` to reach the Forchheimer regime.

---

## Pipeline: Steps, Inputs, and Outputs

### Step 0 — Dimensional Analysis (Buckingham-Pi)

**Driver:** `pydimension.data_preprocessing.DataPreprocessor` — same null-space +
SymPy primitive-integer reduction used by every other Stage1 example. The
script writes a hand-checked `dimension_matrix.csv` to `_da_repo/` and calls
`DataPreprocessor.process_with_dimensional_analysis()`. An inline
`compute_pi_basis()` (scipy + SymPy) is retained as a fallback and can be
forced via `--no-repo-da`.

**Input:** 3×6 dimension matrix (M, L, T × 6 variables; phi already dimensionless).

**Output — 3 Pi groups (DataPreprocessor):**
```
π1 = dP_L · v⁻³ · μ · ρ⁻²        (= f / Re_p² in the f-Re_p basis)
π2 = dP_L · v⁻¹ · μ⁻¹ · d²       (Darcy form)
π3 = φ
```

Any null-space basis is valid — the canonical pair `Re_p = ρvd/μ` and
`f = dP_L·d/(ρv²)` is a different choice of basis vectors in the same span.

**Verification:** Project known `f` and `Re_p` exponent vectors onto the
discovered null-space → both give cos = +1.0000 ✓ (confirming `f` and `Re_p`
lie exactly in the span of the discovered Pi groups).

---

### Step 1 — Normalisation

- `log10(Re_p)` and `phi` min-max scaled → Pi features for Step 2
- `Re_p` and `phi` **values** geometric-mean-centred (a purely multiplicative
  rescaling, no min-max) → Step 3 input
- `log10(f)` (NN target) min-max scaled — f spans 5 orders of magnitude, so
  log-transformation is essential

---

### Step 2 — Latent Dimension Discovery

**Input:** 2 Pi features: `[log10(Re_p), phi]`.

**Encoder:** Multilayer MLP `[2 → 64 → 32 → k]`. Sweep `k = 1, 2`.

**Expected (for deep Darcy regime):**
- k = 1: R² high but not perfect (Re_p alone misses phi dependence)
- k = 2: R² → 1.0 (Re_p and phi both needed)

---

### Step 3 — Symmetry Type Identification

**Input:** The 2 Pi **values** `(Re_p, phi)`, geometric-mean-centred.

Three competing single-layer encoders test power-law (`log|Pi|`),
additive (`Pi`), and quadratic (`Pi²`) symmetries — now acting on
dimensionless quantities, so the scaling encoder's internal log sees
centred log-Pi coordinates directly.

**Expected (physics):** Scaling should win, because the Darcy law
`f ∝ 1/Re_p` is a pure power-law.

**Observed (Pi-space Step 3):**

```
scaling        : 0.000473  ← winner
translational  : 0.001909
rotational     : 0.002972
Loss gap: 4.0×
```

Scaling now wins decisively. The earlier raw-X Step 3 was volatile
(winner flipped run-to-run; see the historical caveat in Observed Results)
because only 3 of 6 physical columns actually varied and min-max scaling
distorted the multiplicative structure. In Pi space both problems vanish.

**Discovered direction vs Darcy:** in the deep-Darcy regime
`f ≈ 150·(1−φ)²/(Re_p·φ³)`, so the local log-slope reference is
`[∂logf/∂logRe, ∂logf/∂logφ] = [−1, −5.39]` at `φ̄ = 0.544`. The winning
encoder row satisfies **cos(W, Darcy reference) = ±0.993**.

---

### Step 4 — Generator Extraction

With `k* = 1` and 2 Pi inputs, there is `2 − 1 = 1` null-space generator
in log-Pi space. (The old raw-X caveat about the constant `d` column no
longer applies — both Pi inputs vary meaningfully.)

---

### Step 5 — Physical Interpretation

**Actual generator (Pi space):**

| Generator | Trade-off | Meaning |
|---|---|---|
| G1 | `Re_p` × exp(−0.96ε), `phi` × exp(+0.29ε) | Higher porosity lowers friction; a lower Reynolds number raises it back — moving along this direction keeps `f` constant |

---

## Output Files

All outputs go to `output_porous_media_lbm_symmetry/`.

| File | Contents |
|---|---|
| `lbm_ergun_collapse.png` | f·φ³/(1−φ) vs Re_p/(1−φ), coloured by porosity, with textbook curve overlaid |
| `lbm_pi_candidates.png` | Pi-basis heatmap + scatter of log₁₀(f) vs each log₁₀(Πₖ) |
| `lbm_symmetry_discovery.png` | 2-panel: symmetry-type bars + latent-dim R² curve |
| `run.log` | Full stdout log of the pipeline run |
| `_da_repo/dimension_matrix.csv` | Hand-checked integer dimension matrix fed to `DataPreprocessor` |
| `_da_repo/data/basis_vectors.csv` | Primitive integer Pi-group exponent vectors emitted by `DataPreprocessor` |
| `_da_repo/data/afterDA_data.csv` | Normalised Pi values per row produced by `DataPreprocessor` |

---

## Observed Results (committed run)

The committed output figures were produced with reduced training
(`--latent-epochs 300 --sym-epochs 600 --n-restarts 3 --seed 42`) and
locked threads + hash seed for bit-reproducibility:
```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0
```
Step 0 is driven by `pydimension.data_preprocessing.DataPreprocessor`.

| Aspect | Observed |
|---|---|
| Pi-basis driver | `DataPreprocessor.process_with_dimensional_analysis()` ✓ |
| Pi groups discovered | 3 — `π1 = dP_L·v⁻³·μ·ρ⁻²`, `π2 = dP_L·v⁻¹·μ⁻¹·d²`, `π3 = φ` |
| Known `f` exponents in Pi span | **cos = +1.0000 ✓** |
| Known `Re_p` exponents in Pi span | **cos = +1.0000 ✓** |
| Latent dimension `k*` | **1** (R² ≈ 0.997 at k=1) |
| Symmetry type (Pi-space Step 3) | **Scaling ✓** (MSE 0.000473) — winner by **4.0×** over translational (0.00191) and rotational (0.00297) |
| Discovered direction | cos(W, Darcy `[−1, −5.39]`) = ±0.993 |
| Generators | 1 direction in log-(Re_p, φ) space: `Re_p` × exp(−0.96ε), `φ` × exp(+0.29ε) |

**Stable physics (Step 0):** The Buckingham-Pi reduction is fully
reproducible. `DataPreprocessor` returns the same three primitive integer
Pi groups every run, and the known `f` and `Re_p` exponent vectors both
project onto the discovered span with `cos = +1.0000`. **This is the
physically meaningful result of the example.**

**Ergun collapse:** All 4 porosity bins collapse onto a single curve with
slope = −1 (pure Darcy), running **~30% below** the textbook `150/x` line.
This systematic offset (mean `f / f_ergun ≈ 0.66`) is consistent across all
data and indicates the LBM under-predicts drag relative to the textbook
empirical constant — likely a combination of single-packing-realisation
effects, `d/Δx = 10` resolution, and bounce-back boundary slip.

**Why `k* = 1` instead of the expected `k* = 2`?** Because in the pure
Darcy regime `f ≈ 150·(1−φ)²/(φ³·Re_p)`, and `log(f)` varies by ~5 orders
of magnitude from Re_p but only ~0.65 from the φ factor across this
porosity range. So `Re_p` alone explains 99.7% of the variance —
`φ` only contributes a small logarithmic correction. To force `k* = 2`,
we'd need a wider φ range or higher Re_p where the inertial Forchheimer
term `1.75·(1−φ)/φ³` becomes important.

**Scaling wins decisively across all six seeds:**

| Seed | Winner | scaling MSE | trans MSE | rot MSE | gap |
|---|---|---|---|---|---|
| 42  | **scaling** | **0.000543** | 0.002638 | 0.004735 | 4.9× |
| 0   | **scaling** | **0.003283** | 0.009857 | 0.010836 | 3.0× |
| 1   | **scaling** | **0.003573** | 0.007249 | 0.009900 | 2.0× |
| 2   | **scaling** | **0.002314** | 0.005995 | 0.012743 | 2.6× |
| 7   | **scaling** | **0.001862** | 0.005840 | 0.005315 | 2.9× |
| 100 | **scaling** | **0.003560** | 0.009357 | 0.010138 | 2.6× |

This is the physically expected result: Darcy's law `f ∝ 1/Re_p` is a
pure power-law, so `log f` is linear in `log(ρ·v·d/μ)`, which is
exactly the family of relationships the scaling encoder
`z = W · log|X|` is designed to recover.

**Why earlier runs got "rotational":** The Stage1 scaling encoder
applies `log(|X|.clamp(min=0.1))` internally. The clamp threshold of
0.1 is calibrated for **raw multiplicatively-meaningful X** (positive
values of order unity, spanning a few orders of magnitude). If we
min-max-normalise X into `[0, 1]` first — the default Stage1
preprocessing for the other two examples — many values land below
0.1 and get pinned by the clamp, destroying the multiplicative signal
that the scaling encoder needs. The rotational (`X²`) and
translational (`X`) encoders are bijective on `[0, 1]` so they keep
the full input information, and a flexible MLP decoder approximates
the needed `log()` internally — letting them appear to win on raw fit
MSE even though their feature maps don't encode the underlying
invariance.

The fix in this example (current form): **Step 3 runs on the Pi values
themselves** — `(Re_p, φ)`, each column divided by its geometric mean
(per-column geometric mean exactly 1.0), passed directly to
`identify_symmetry` with no min-max. This keeps the multiplicative
structure intact for the clamp regime the log encoder was designed for,
and additionally removes the three non-varying physical columns from the
encoder entirely. Result: a stable scaling win (4.0×) and a weight
vector aligned with the Darcy slope.

**Reproducibility:** lock the three BLAS thread vars and
`PYTHONHASHSEED` so the run is bit-reproducible. Without
`PYTHONHASHSEED=0`, Python's string hash randomisation feeds different
torch seeds into `identify_symmetry` each invocation (it computes
`seed + hash(sym_type) % 1000 + restart * 37`).

---

## Usage

```bash
cd projects/20260912_Stage1_Prokash/Examples/porous_media_lbm_symmetry

# Reproducible run (matches committed run.log)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python discover_symmetry.py --data dataset_lbm_porous.csv \
    --latent-epochs 300 --sym-epochs 600 --n-restarts 3 --seed 42

# Default run (multi-threaded, faster, but Step 3 winner is non-deterministic)
python discover_symmetry.py --data dataset_lbm_porous.csv

# Deeper encoder
python discover_symmetry.py --data dataset_lbm_porous.csv --encoder-hidden 128 64 32

# Longer training for stable symmetry detection
python discover_symmetry.py --data dataset_lbm_porous.csv --sym-epochs 3000

# Different output directory
python discover_symmetry.py --data dataset_lbm_porous.csv --output-dir my_output
```

Default training budget: `--latent-epochs 600`, `--sym-epochs 1500`,
`--n-restarts 3`, `--encoder-hidden 64 32`.

---

## File Organisation

```
PyDimension/
├── pydimension/                              ← auto-discovered
└── projects/20260912_Stage1_Prokash/
    ├── preprocessing/
    ├── intrinsic_coordinate/
    ├── symmetry_discovery/
    └── Examples/porous_media_lbm_symmetry/
        ├── PLAN.md                           ← full design plan
        ├── README.md                         ← this file
        ├── discover_symmetry.py              ← Stage1 pipeline script
        └── dataset_lbm_porous.csv            ← 144-row LBM dataset (r ∈ {3, 5})
```

Dependencies: `torch`, `numpy`, `scipy`, `sympy`, `matplotlib`, `seaborn`.
Install with `pip install -r requirements.txt` from the repo root.

---

## Full-Range Extension: Viscous + Inertial (`dataset_combined_ergun.csv`)

The committed LBM data only probes the **viscous** branch (Re_p < 1e-3), so
the inertial Forchheimer term (1.75) is invisible. To test what the pipeline
does when the data spans the **whole** Ergun curve,
`generate_combined_dataset.py` fills every Re_p region the LBM data does not
cover: it adds 360 **synthetic** rows over a wide range Re_p ≈ 1e-3 – 1e6
(the transition knee at Re_p ≈ 85 *and* the full inertial plateau), drawn from
the **textbook Ergun equation** (`150/X + 1.75`) with 5 % multiplicative noise
and physically self-consistent six-input rows. Combined with the 144 real LBM
rows this is a **504-row** set covering the full curve. `plot_full_ergun_range.py`
draws the textbook master curve with the synthetic points on it and the **real
LBM points marked distinctly** — the LBM data sits ~0.65× below the textbook
viscous branch (`A_eff ≈ 97.5`, LBM gives ~35 % lower drag), so the offset is
shown explicitly (`combined_ergun_full_range.png`).

**Run it (GPU):**

```bash
python run_fullrange_check.py               # generate -> plot -> pipeline, tees a log
python run_fullrange_check.py --noise 0.2   # more synthetic scatter (~22%)
```

The synthetic scatter is the log-normal noise width `--noise` (default
0.05 ≈ 5 %; 0.2 ≈ 22 %, 0.5 ≈ 65 %), forwarded to
`generate_combined_dataset.py`, which also exposes `--re-jitter`,
`--n-re`, `--re-min-exp`, `--re-max-exp`, and `--seed`.

This writes `output_porous_fullrange/fullrange_check_full.log` (share it back).
Individual steps:

```bash
python generate_combined_dataset.py     # -> dataset_combined_ergun.csv (504 rows)
python plot_full_ergun_range.py         # -> combined_ergun_full_range.png
python discover_symmetry.py --data dataset_combined_ergun.csv \
    --seed 42 --output-dir output_porous_fullrange
```

**Expected finding.** An earlier variant of this experiment (synthetic scaled
to the LBM effective constant) already showed the key behaviour, and the
textbook version is physically the same two-term sum, so the same pattern is
expected: the clean scaling symmetry that governs the Darcy branch **does not
survive** extension to the full curve. A single power law (`f ∝ 1/Re_p`, deep
Darcy) is scale-invariant, but the **sum of two power laws**
(`f = A/Re_p + B` in collapse coordinates) is *not* — no global rescaling (or
shift) of `(Re_p, φ)` preserves `f` across both regimes. In the earlier runs
the winning symmetry margin fell from **3.7×** (LBM only, clean scaling) to
~**1.0–1.8×** on the full range, the winning **type flipped** scaling →
translational, and the recovered generator stopped aligning with the Darcy
scaling direction (cos −0.98 → −0.32). The friction-factor fit stays high
(R² ≈ 0.997) because the flexible decoder still represents the curved law — it
is the **symmetry structure**, not the regression, that degrades. Takeaway:
the method cleanly identifies a symmetry **only within a single power-law
regime**; across a crossover it correctly reports a weak, unstable symmetry.
Fill in this section's exact numbers from your `fullrange_check_full.log`.

---

## Future Work

1. **Add particle-size variation** — re-run LBM with r ∈ {4, 6, 8} to fully
   constrain the d-direction generator.
2. **Reach Forchheimer regime** — add `delta_p ∈ {1e-3, 1e-2}` to push
   Re_p > 10 and expose the inertial 1.75 term.
3. **Investigate the f/f_ergun ≈ 0.66 offset** — compare against the
   Macdonald–Ergun correlation (uses 180 instead of 150 for natural particles)
   to see if your packing follows a different empirical constant.
