# Data-Driven Discovery of Translational Symmetry in Concrete Compressive Strength

## Abstract

We apply the PyDimension Stage&nbsp;1 symmetry-discovery pipeline to the UCI
Concrete Compressive Strength dataset (Yeh, 1998; 1030 samples, 8 mix-design
inputs) to test whether the governing functional form is consistent with an
**additive (translational)** symmetry. A multilayer perceptron (MLP) encoder
with hidden widths `[64, 32]` is trained jointly with a paired decoder. The
encoder consumes the eight standardised mix-design inputs **directly**
(`raw_input=True`, no `[X, X², log|X|]` augmentation); the MLP is expected
to compose any non-linearity it needs internally. The intrinsic latent
dimension is identified via held-out reconstruction performance, and the
symmetry type is determined by a competitive encoder-training step over
translational, rotational, and scaling candidates. The translational
candidate is selected with a **3.3&times;** validation-MSE gap over the
next-best (scaling) candidate, and **six** independent Lie-algebra
generators are extracted. Each generator corresponds to a physically
interpretable **mix substitution** that preserves compressive strength.

## 1. Problem Statement

Concrete compressive strength is governed predominantly by additive
mix-proportion relationships:

- Total binder mass: `m_binder = m_cement + m_slag + m_fly_ash`
- Water-to-binder ratio: `w/b = m_water / m_binder`
- Strength model: `σ_c ≈ f(a₁·m_cement + a₂·m_slag + a₃·m_fly_ash + a₄·m_water + …)`

Under the additive ansatz, the output depends on the data only through a
linear combination `z = W x`, so any shift `x → x + ε g` with `W g = 0`
leaves `σ_c` invariant. The null space of `W` therefore parametrizes the
**translational symmetry generators** of the strength surface.

## 2. Dataset

| Variable | Symbol | Units | Range |
|---|---|---|---|
| Cement | `m_c` | kg/m³ | 102 – 540 |
| Blast-furnace slag | `m_s` | kg/m³ | 0 – 359 |
| Fly ash | `m_f` | kg/m³ | 0 – 200 |
| Water | `m_w` | kg/m³ | 122 – 247 |
| Superplasticizer | `m_p` | kg/m³ | 0 – 32 |
| Coarse aggregate | `m_{ca}` | kg/m³ | 801 – 1145 |
| Fine aggregate | `m_{fa}` | kg/m³ | 594 – 993 |
| Age | `t` | days | 1 – 365 |
| **Compressive strength** | `σ_c` | MPa | 2.3 – 82.6 |

*Source:* UCI Machine Learning Repository, dataset&nbsp;#165
([archive.ics.uci.edu/dataset/165](https://archive.ics.uci.edu/dataset/165)),
1030 samples.

## 3. Methodology

The pipeline implements five sequential stages:

1. **Normalization.** Standard scaling (zero mean, unit variance) of the
   eight inputs and of the output.
2. **Intrinsic-dimension discovery.** A latent-bottleneck autoencoder is
   trained for `k ∈ {1, 2, 3, 4}`. The encoder is a multilayer perceptron
   with hidden widths `[64, 32]` and `Tanh` activations operating on the
   raw standardised inputs (`raw_input=True`, no `[X, X², log|X|]`
   augmentation); the decoder is a paired MLP of matching capacity. Each
   `k` is repeated over `n_restarts = 3` random seeds and 600 epochs, and
   the latent dimension minimising the held-out reconstruction MSE is
   selected.
3. **Symmetry-type identification.** Three competing encoder families are
   trained against the Step&nbsp;2 decoder:
   - **Translational:** `z = W x`,
   - **Scaling:** `z = W · log|x|`,
   - **Rotational:** `z = W · ½ x⊙x` (quadratic).

   Each is trained for 1500 epochs with `n_restarts = 3`. The candidate
   with the lowest held-out MSE is declared the winner.
4. **Generator extraction.** For the translational winner, the Lie-algebra
   generators are the null-space basis of `W`: any vector `g` with `W g = 0`
   defines an infinitesimal shift `x → x + ε g` that preserves `σ_c`.
5. **Physical interpretation.** Each generator is rendered as a signed
   list of mix components, identifying the substitution it represents
   (e.g. "increase superplasticizer while decreasing fly ash").

All randomness is seeded (`seed = 42`); the complete configuration is
captured in `output_concrete_symmetry/run.log`.

## 4. Results

### 4.1 Latent dimension

The intrinsic latent dimension is `k = 2`. The held-out coefficient of
determination peaks at `R² = 0.892` for `k = 2` and degrades slightly
for both lower and higher `k`:

| `k` | `R²_train` | `R²_test` | MSE |
|---|---|---|---|
| 1 | 0.952 | 0.879 | 0.1123 |
| 2 | 0.953 | **0.892** | **0.0999** |
| 3 | 0.959 | 0.885 | 0.1066 |
| 4 | 0.957 | 0.875 | 0.1157 |

### 4.2 Symmetry type

Competitive training cleanly selects the translational candidate:

| Symmetry candidate | Held-out MSE |
|---|---|
| **translational** | **0.1514** |
| scaling | 0.4993 |
| rotational | 0.6206 |

The translational candidate beats the second-best (scaling) candidate by a
factor of **3.3&times;** in validation MSE, confirming that the governing
combination of inputs is additive rather than multiplicative.

### 4.3 Generators

With `n = 8` inputs and `k = 2` latent directions, there are
`n − k = 6` independent translational generators. The dominant components
of each are listed below (only `|g_j| > 0.05` shown):

| Generator | Dominant components | Physical reading |
|---|---|---|
| `g₁` | Fly Ash (+0.99), Cement (−0.08), Age (−0.06), Slag (−0.06) | Replace cement with fly ash at fixed strength |
| `g₂` | Water (+0.85), Slag (+0.48), Superplast. (+0.18), Age (−0.08) | Co-vary water and slag while reducing age |
| `g₃` | Superplast. (+0.80), Slag (−0.55), Water (+0.18), Cement (−0.07) | Replace slag/cement with superplasticizer + water |
| `g₄` | Coarse Agg. (+0.96), Slag (+0.22), Age (−0.10), Water (−0.08) | Replace water/age with coarse aggregate + slag |
| `g₅` | Fine Agg. (+0.93), Slag (+0.31), Superplast. (+0.13), Age (−0.12) | Replace water/age with fine aggregate + slag |
| `g₆` | Cement (−0.79), Slag (+0.39), Superplast. (+0.27), Water (−0.24) | Replace cement + water with slag + superplasticizer |

Each generator is a constant-strength direction in mix-design space:
moving the composition along `g_i` (within physical limits) leaves the
predicted compressive strength unchanged.

### 4.4 Figure

`output_concrete_symmetry/concrete_symmetry_discovery.png` reports:
(left) the learned latent embedding coloured by `σ_c`, showing a clear
strength gradient along the first latent direction; (right) the
validation-MSE bar chart of the three competing symmetry candidates.

## 5. Reproducibility

### Environment

```bash
pip install torch numpy matplotlib pandas openpyxl xlrd
```

### Data

Download the UCI Concrete Compressive Strength dataset
([dataset #165](https://archive.ics.uci.edu/dataset/165)) and place
`Concrete_Data.xls` (or a CSV export) in this directory.

### Reproduce the reported results

```bash
python discover_symmetry.py \
    --data Concrete_Data.xls \
    --seed 42 \
    --latent-epochs 600 \
    --sym-epochs 1500 \
    --n-restarts 3
```

The script defaults to `--encoder-hidden 64 32` and `raw_input=True`
(no `[X, X², log|X|]` augmentation at Step 2), so no extra flags are
required.

Output is written to `output_concrete_symmetry/`:

- `concrete_symmetry_discovery.png` — two-panel summary figure.
- `run.log` — full console transcript (config, per-`k` metrics, symmetry
  losses, generator decomposition).

A baseline run with a single-layer linear encoder is obtained by omitting
`--encoder-hidden`.

## 6. Discussion

The translational fingerprint recovered here is consistent with the
domain understanding that compressive strength is governed by
*water-to-binder ratio* and *total binder mass*, both of which are linear
combinations of the mix components. The multilayer encoder lifts the
restrictive single-direction assumption of a linear encoder and resolves
a two-dimensional latent manifold, while still preserving the
translational character of the symmetry. Disabling the
`[X, X², log|X|]` augmentation (`raw_input=True`) lets the MLP discover
the correct nonlinear combinations on its own; the resulting six
generators provide an interpretable, data-driven catalogue of
strength-preserving mix substitutions that can guide constrained
mix-design optimisation (e.g. supplementary cementitious material
substitution at fixed target strength).

## 7. Dimensionless (Buckingham-Pi) Variant

`discover_symmetry_dimensionless.py` repeats the experiment on a
non-dimensionalized representation instead of raw standardized kg/m³
inputs. All seven mix quantities share the dimension [M L⁻³], so ratios
by the total binder mass `b = cement + slag + fly ash` are dimensionless.
Following Yeh (1998): Table 7 shows his w/b convention counts the
superplasticizer dose as water, and Table 6 (random-split experiments
R1–R4, averaged) provides a regression baseline used to form a residual
target:

- **Features:** `w/b = (water+SP)/b`, `flyash/b`, `slag/b`, `SP/b`,
  `coarse/b`, `fine/b`, `t/28 d` — seven pure numbers, no transforms
  beyond the ratios themselves.
- **Baseline:** `σ_ideal = 13.83·(w/b)^(−1.269)·(0.268·ln t + 0.136)` MPa
  (the `ln t` here is Yeh's published fitted form, kept verbatim).
- **Target:** `y = σ_c / σ_ideal` — dimensionless strength residual.

### Results (seed 42, same pipeline settings as Section 5)

| Quantity | Raw-input run (Sec. 4) | Dimensionless run |
|---|---|---|
| Target | standardized σ_c | σ_c/σ_ideal |
| Yeh baseline R² (no ML) | — | **0.762** (paper: ≈0.77) |
| Mean σ_c/σ_ideal | — | 0.971 ± 0.233 |
| Optimal latent dim | 2 | 3 |
| Held-out R² | 0.892 (of σ_c) | 0.555 (of the *residual*) |
| Symmetry winner | translational (3.3×) | **translational (1.4×)** |
| Generators | 6 | 4 |

The translational fingerprint survives the change of coordinates: the
strength residual is additive in the binder-referenced mix ratios. The
R² values are not comparable across columns — the dimensionless run
models only the variance *left over* after the analytic baseline has
removed the dominant w/b and age effects. The four residual generators
describe strength-preserving substitutions in ratio space, e.g.
generator 2 trades w/b against slag fraction and age
(`w/b: −0.78, slag/b: +0.36, t/28: +0.27`) and generator 4 exchanges
slag for fly ash (`flyash/b: +0.55, slag/b: −0.70`). Note that the
untransformed `t/28` feature is strongly right-skewed (0.036–13) and
the raw ratio target compresses the low-strength end relative to a
log residual, which lowers the residual R² and narrows the symmetry
margin compared to a logarithmic variant of the same experiment.

Output is written to `output_concrete_dimensionless/`
(`concrete_symmetry_dimensionless.png`, `run.log`).

## 8. References

1. I-C. Yeh, "Modeling of strength of high-performance concrete using
   artificial neural networks," *Cement and Concrete Research*,
   **28**(12), 1797–1808, 1998.
2. UCI Machine Learning Repository, Concrete Compressive Strength
   dataset #165. <https://archive.ics.uci.edu/dataset/165>
3. PyDimension Stage&nbsp;1: symmetry-aware dimensional-analysis pipeline
   (this repository, `projects/20260912_Stage1_Prokash/`).
