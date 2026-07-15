# Data-Driven Discovery of Translational Symmetry in Concrete Compressive Strength Using Dimensionless Variables

## Abstract

We apply the PyDimension Stage&nbsp;1 symmetry-discovery pipeline to the UCI
Concrete Compressive Strength dataset (Yeh, 1998; 1030 samples, 8 mix-design
inputs) using a **dimensionless (Buckingham-Pi) representation**. All seven
mix quantities share the dimension [M&nbsp;L⁻³], so they are reduced to six
ratios by the total binder mass; curing age enters as the logarithm of the
dimensionless age ratio `ln(t/28 d)`. The target is the dimensionless
strength residual `σ_c/σ_ideal`, where `σ_ideal` is the regression baseline
published in the source paper (Yeh, 1998, Table&nbsp;6). The frozen baseline
alone explains R²&nbsp;=&nbsp;0.762 of the strength variance; the pipeline
then models the residual with a multilayer-perceptron autoencoder (hidden
widths `[64, 32]`, `raw_input=True`). The intrinsic latent dimension is
identified as `k = 4`, and competitive encoder training selects the
**translational** symmetry candidate with a **1.9×** validation-MSE gap over
the next-best (rotational) candidate. Three independent Lie-algebra
generators are extracted, each a physically interpretable
strength-preserving substitution in mix-ratio space.

## 1. Problem Statement

Concrete compressive strength is governed predominantly by additive
mix-proportion relationships expressed through dimensionless ratios:

- Total binder mass: `b = m_cement + m_slag + m_fly_ash`
- Water-to-binder ratio: `w/b`
- Strength model: `σ_c ≈ f(a₁·π₁ + a₂·π₂ + …)` for dimensionless groups `πᵢ`

Under the additive ansatz, the output depends on the data only through a
linear combination `z = W π`, so any shift `π → π + ε g` with `W g = 0`
leaves the strength residual invariant. The null space of `W` parametrizes
the **translational symmetry generators** of the residual strength surface.

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

## 3. Non-Dimensionalization

### 3.1 Input features (Buckingham Pi)

The seven mass quantities all carry the dimension [M L⁻³]; by the
Buckingham-Pi theorem they reduce to six dimensionless ratios with respect
to one reference quantity. Following Yeh (1998), the reference is the
**total binder mass** `b = m_c + m_s + m_f`, and — per the convention
reverse-engineered from Table&nbsp;7 of that paper — the superplasticizer
dose is counted as water in the w/b numerator:

| Feature | Definition |
|---|---|
| `π₁` | `w/b = (m_w + m_p) / b` |
| `π₂` | `m_f / b` (fly-ash replacement fraction) |
| `π₃` | `m_s / b` (slag replacement fraction) |
| `π₄` | `m_p / b` (superplasticizer dosage) |
| `π₅` | `m_{ca} / b` |
| `π₆` | `m_{fa} / b` |
| `π₇` | `ln(t / 28 d)` (dimensionless age) |

Age carries the only [T] dimension among the inputs and cannot be
non-dimensionalized against other columns; it is referenced to the
industry-standard 28-day curing age. The logarithm is applied to the age
ratio only: it symmetrizes the heavily skewed 1–365-day range around
`π₇ = 0` at 28 days and matches the logarithmic age kinetics of the
baseline model. The strength is left untransformed.

### 3.2 Baseline and target

Yeh (1998) fitted the regression `f′c = a·(w/b)^β·(c·ln t + d)` to his
database; averaging the coefficients of the four random-split experiments
(Table&nbsp;6, rows R1–R4) gives the frozen baseline

```
σ_ideal = 13.83 · (w/b)^(−1.269) · (0.268·ln t + 0.136)   [MPa, t in days]
```

The learning target is the **dimensionless strength residual**

```
y = σ_c / σ_ideal
```

Because `σ_ideal` already carries the dominant w/b and age effects, the
pipeline models only the residual chemistry (SCM substitution,
superplasticizer, aggregates). The coefficients come from the 1998
publication, not from this dataset, so no train/test leakage is possible.

On the full 1030-row dataset the frozen baseline alone achieves
**R² = 0.762** (the paper reports ≈0.77 on its 727 records), and the
residual is well-centred: `mean(σ_c/σ_ideal) = 0.971 ± 0.233`.

## 4. Methodology

The pipeline implements six sequential stages:

1. **Non-dimensionalization.** Construction of `π₁ … π₇` and
   `y = σ_c/σ_ideal` as defined in Section 3.
2. **Normalization.** Standard scaling (zero mean, unit variance) of the
   seven dimensionless features and of the residual target.
3. **Intrinsic-dimension discovery.** A latent-bottleneck autoencoder is
   trained for `k ∈ {1, 2, 3, 4}`. The encoder is a multilayer perceptron
   with hidden widths `[64, 32]` and `Tanh` activations operating on the
   raw standardised features (`raw_input=True`, no `[X, X², log|X|]`
   augmentation); the decoder is a paired MLP of matching capacity. Each
   `k` is repeated over `n_restarts = 3` random seeds and 600 epochs, and
   the latent dimension minimising the held-out reconstruction MSE is
   selected.
4. **Symmetry-type identification.** Three competing encoder families are
   trained against the Step&nbsp;3 decoder:
   - **Translational:** `z = W π`,
   - **Scaling:** `z = W · log|π|`,
   - **Rotational:** `z = W · ½ π⊙π` (quadratic).

   Each is trained for 1500 epochs with `n_restarts = 3`. The candidate
   with the lowest held-out MSE is declared the winner.
5. **Generator extraction.** For the translational winner, the Lie-algebra
   generators are the null-space basis of `W`: any vector `g` with
   `W g = 0` defines an infinitesimal shift `π → π + ε g` that preserves
   the strength residual.
6. **Physical interpretation.** Each generator is rendered as a signed
   list of mix ratios, identifying the substitution it represents.

All randomness is seeded (`seed = 42`); the complete configuration is
captured in `output_concrete_dimensionless/run.log`.

## 5. Results

### 5.1 Latent dimension

The intrinsic latent dimension of the residual is `k = 4`:

| `k` | `R²_train` | `R²_test` | MSE |
|---|---|---|---|
| 1 | 0.729 | 0.488 | 0.4645 |
| 2 | 0.747 | 0.505 | 0.4493 |
| 3 | 0.738 | 0.545 | 0.4134 |
| 4 | 0.749 | **0.556** | **0.4032** |

The R² values refer to the *residual* `σ_c/σ_ideal`, i.e. to the variance
left over after the analytic baseline has removed the dominant w/b and
age effects.

### 5.2 Symmetry type

Competitive training selects the translational candidate:

| Symmetry candidate | Held-out MSE |
|---|---|
| **translational** | **0.3079** |
| rotational | 0.5985 |
| scaling | 0.6380 |

The translational candidate beats the second-best (rotational) candidate
by a factor of **1.9×** in validation MSE: the strength residual is
additive in the binder-referenced mix ratios.

### 5.3 Generators

With `n = 7` dimensionless features and `k = 4` latent directions, there
are `n − k = 3` independent translational generators (components with
`|g_j| > 0.05` shown):

| Generator | Dominant components | Physical reading |
|---|---|---|
| `g₁` | Slag/b (+0.62), SP/b (+0.49), CoarseAgg/b (+0.42), FlyAsh/b (−0.29), ln(t/28) (+0.25), FineAgg/b (−0.22) | Replace fly ash and fine aggregate with slag, superplasticizer, and coarse aggregate at longer curing |
| `g₂` | FineAgg/b (+0.75), SP/b (+0.61), CoarseAgg/b (−0.24) | Exchange coarse for fine aggregate with added superplasticizer |
| `g₃` | w/b (−0.56), FlyAsh/b (+0.44), Slag/b (−0.47), SP/b (+0.30), CoarseAgg/b (+0.33), ln(t/28) (+0.25) | Trade lower w/b and slag against fly ash, superplasticizer, and curing age |

Each generator is a constant-residual direction in mix-ratio space:
moving the composition along `g_i` (within physical limits) leaves the
predicted strength residual `σ_c/σ_ideal` unchanged.

### 5.4 Figure

`output_concrete_dimensionless/concrete_symmetry_dimensionless.png`
reports: (left) measured strength against the Yeh baseline `σ_ideal`
with the 1:1 line (R² = 0.762); (centre) the learned latent embedding
coloured by the strength residual; (right) the validation-MSE bar chart
of the three competing symmetry candidates.

## 6. Reproducibility

### Environment

```bash
pip install torch numpy scipy matplotlib pandas openpyxl xlrd
```

### Data

Download the UCI Concrete Compressive Strength dataset
([dataset #165](https://archive.ics.uci.edu/dataset/165)) and place
`Concrete_Data.xls` (or a CSV export) in this directory.

### Reproduce the reported results

```bash
python discover_symmetry_dimensionless.py \
    --data Concrete_Data.xls \
    --seed 42 \
    --latent-epochs 600 \
    --sym-epochs 1500 \
    --n-restarts 3
```

The script defaults to `--encoder-hidden 64 32` and `raw_input=True`, so
no extra flags are required.

Output is written to `output_concrete_dimensionless/`:

- `concrete_symmetry_dimensionless.png` — three-panel summary figure.
- `run.log` — full console transcript (config, baseline fit, per-`k`
  metrics, symmetry losses, generator decomposition).

## 7. Discussion

Non-dimensionalization factorizes the problem into a citable analytic
baseline and a learned dimensionless correction. The binder-referenced
ratios quotient out the overall "scale the whole mix" direction
analytically, and the Yeh baseline removes the two dominant physical
effects (w/b and age), so the network's entire capacity is spent on the
residual chemistry. The translational fingerprint recovered on these
coordinates confirms that the residual strength surface is governed by
additive combinations of the mix ratios. The three generators provide an
interpretable, data-driven catalogue of strength-preserving mix
substitutions — e.g. supplementary-cementitious-material exchange
(`g₃`: fly ash for slag at reduced w/b) or aggregate grading shifts
compensated by superplasticizer (`g₂`) — that can guide constrained
mix-design optimisation at a fixed target strength.

## 8. References

1. I-C. Yeh, "Modeling of strength of high-performance concrete using
   artificial neural networks," *Cement and Concrete Research*,
   **28**(12), 1797–1808, 1998.
2. UCI Machine Learning Repository, Concrete Compressive Strength
   dataset #165. <https://archive.ics.uci.edu/dataset/165>
3. PyDimension Stage&nbsp;1: symmetry-aware dimensional-analysis pipeline
   (this repository, `projects/20260912_Stage1_Prokash/`).
