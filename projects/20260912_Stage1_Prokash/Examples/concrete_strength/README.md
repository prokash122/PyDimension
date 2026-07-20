# Data-Driven Discovery of Translational Symmetry in Concrete Compressive Strength Using Dimensionless Variables

## Abstract

We apply the PyDimension Stage&nbsp;1 symmetry-discovery pipeline to the UCI
Concrete Compressive Strength dataset (Yeh, 1998; 1030 samples, 8 mix-design
inputs) using a **binder-referenced dimensionless representation**. Every
mix mass carries the dimension [M&nbsp;L⁻³], so simply dividing it by the
total binder mass `b = cement + slag + fly ash` yields a dimensionless
ratio — the standard concrete-science normalization (water/binder ratio,
SCM replacement fractions, aggregate/binder ratios), with no Buckingham-Pi
bookkeeping required. Curing age enters as the logarithm of the
dimensionless age ratio `ln(t/28 d)`. The water/binder ratio is the
**literal** `w/b = m_w / b` (superplasticizer keeps its own ratio and is
not folded into the water term). The target is the dimensionless
strength residual `σ_c/σ_ideal`, where `σ_ideal` is the regression baseline
published in the source paper (Yeh, 1998, Table&nbsp;6), evaluated with the
same literal `w/b`. The frozen baseline alone explains
R²&nbsp;=&nbsp;0.682 of the strength variance; the pipeline then models the
residual with a multilayer-perceptron autoencoder (hidden widths
`[64, 32]`, `raw_input=True`). The intrinsic latent dimension is
identified as `k = 4` (an interior optimum of a search over
`k ∈ {1, …, 6}`), and competitive encoder training selects the
**translational** symmetry candidate with a **1.5×** validation-MSE gap over
the next-best (rotational) candidate. With eight features and `k = 4`, the
null space yields **four** Lie-algebra generators — three physically
interpretable strength-preserving substitutions, plus one trivial
direction created by the three binder fractions summing to one. The
publication figure confirms the trained model embodies these generators:
stepping
real mixes along any generator holds the model-predicted dimensionless
strength `σc* = σc/σ_ideal` flat (change ~10⁻⁷), while stepping along
the strength-relevant direction spans the full weak-to-strong range.

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

### 3.1 Input features (division by binder mass)

Every mass quantity carries the dimension [M L⁻³], so dividing it by the
**total binder mass** `b = m_c + m_s + m_f` produces a dimensionless
ratio. No Buckingham-Pi theorem is invoked: the binder mass is simply
chosen as the common reference, exactly as concrete practice already does
with the water/binder ratio, the SCM replacement fractions, and the
aggregate/binder ratios. The water/binder ratio is the **literal**
water-to-binder ratio — superplasticizer is *not* added to the water term;
it retains its own separate ratio `m_p / b`. **All eight** non-dimensional
candidates are kept (the seven mass ratios plus the log-age term):

| Feature | Definition |
|---|---|
| `π₁` | `w/b = m_w / b` (literal water/binder ratio) |
| `π₂` | `m_c / b` (cement fraction of binder) |
| `π₃` | `m_s / b` (slag fraction of binder) |
| `π₄` | `m_f / b` (fly-ash fraction of binder) |
| `π₅` | `m_p / b` (superplasticizer dosage) |
| `π₆` | `m_{ca} / b` |
| `π₇` | `m_{fa} / b` |
| `π₈` | `ln(t / 28 d)` (dimensionless age) |

The three binder fractions sum to one (`m_c/b + m_s/b + m_f/b = 1`), so
the feature set is **rank-deficient by one**: the data lie on a 7-D
hyperplane inside the 8-D feature space. Keeping all three fractions is a
deliberate choice (every raw input maps to its own non-dimensional
candidate), but it means the translational null space carries one extra,
*trivial* "stay on the binder simplex" direction on top of the physical
strength-preserving generators (see Section&nbsp;5.3). Age carries the
only [T] dimension among the inputs and cannot be non-dimensionalized
against other columns; it is
referenced to the industry-standard 28-day curing age. The logarithm is
applied to the age ratio only: it symmetrizes the heavily skewed
1–365-day range around `π₈ = 0` at 28 days and matches the logarithmic
age kinetics of the baseline model. The strength is left untransformed.

### 3.2 Baseline and target

Yeh (1998) fitted the regression `f′c = a·(w/b)^β·(c·ln t + d)` to his
database; averaging the coefficients of the four random-split experiments
(Table&nbsp;6, rows R1–R4) gives the frozen baseline

```
σ_ideal = 13.83 · (w/b)^(−1.269) · (0.268·ln t + 0.136)   [MPa, t in days]
```

where `w/b = m_w / b` is the **same literal water/binder ratio** as `π₁`
(superplasticizer is not counted as water). The learning target is the
**dimensionless strength residual**

```
y = σ_c / σ_ideal
```

Because `σ_ideal` already carries the dominant w/b and age effects, the
pipeline models only the residual chemistry (SCM substitution,
superplasticizer, aggregates). The coefficients come from the 1998
publication, not from this dataset, so no train/test leakage is possible.

On the full 1030-row dataset the frozen baseline alone achieves
**R² = 0.682**, and the residual is well-centred:
`mean(σ_c/σ_ideal) = 0.928 ± 0.223`. The literal `w/b` fits somewhat
below the `(m_w + m_p)/b` convention Yeh reverse-fitted his coefficients
to (R² ≈ 0.76), which is the expected cost of using the plain
water-to-binder ratio rather than folding superplasticizer into the
numerator; the residual chemistry the pipeline then models is
correspondingly a little larger.

## 4. Methodology

The pipeline implements six sequential stages:

1. **Non-dimensionalization.** Construction of `π₁ … π₈` and
   `y = σ_c/σ_ideal` as defined in Section 3.
2. **Normalization.** Standard scaling (zero mean, unit variance) of the
   eight dimensionless features and of the residual target.
3. **Intrinsic-dimension discovery.** A latent-bottleneck autoencoder is
   trained for `k ∈ {1, …, 6}` (with eight features, `k` must stay below
   8 for translational generators to remain). The encoder is a
   multilayer perceptron
   with hidden widths `[64, 32]` and `Tanh` activations operating on the
   raw standardised features (`raw_input=True`, no `[X, X², log|X|]`
   augmentation); the decoder is a paired MLP of matching capacity. Each
   `k` is repeated over `n_restarts = 3` random seeds and 600 epochs, and
   the latent dimension minimising the held-out MSE is selected. On the
   eight-feature set this lands cleanly on `k = 4` (MSE ≈ 0.41 vs ≥ 0.45
   elsewhere); it is also pinned by default (`--latent-dim 4`) for
   reproducibility, since the per-`k` MSEs can tie under other coordinate
   choices — `--latent-dim 0` restores the automatic pick.
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

On the eight-feature set the latent dimension is `k = 4` — a clear
interior optimum (auto-selected, and pinned by default for
reproducibility). Held-out performance peaks at `k = 4` and is markedly
worse on either side:

| `k` | `R²_train` | `R²_test` | MSE |
|---|---|---|---|
| 1 | 0.746 | 0.447 | 0.4698 |
| 2 | 0.782 | 0.462 | 0.4574 |
| 3 | 0.754 | 0.469 | 0.4508 |
| **4** | 0.757 | **0.515** | **0.4122** |
| 5 | 0.742 | 0.453 | 0.4647 |
| 6 | 0.746 | 0.468 | 0.4523 |

The R² values refer to the *residual* `σ_c/σ_ideal`, i.e. to the variance
left over after the analytic baseline has removed the dominant w/b and
age effects. `k = 4` beats every neighbour by ≥ 0.04 in MSE, so unlike
the seven-feature coordinates the optimum is unambiguous here.

### 5.2 Symmetry type

Competitive training selects the translational candidate:

| Symmetry candidate | Held-out MSE |
|---|---|
| **translational** | **0.3740** |
| rotational | 0.5525 |
| scaling | 0.5700 |

The translational candidate beats the second-best (rotational) candidate
by a factor of **1.5×** in validation MSE: the strength residual is
additive in the binder-referenced mix ratios. (The gap and the
runner-up's identity fluctuate between roughly 1.4× and 1.9× across
retrainings because CPU thread scheduling makes the optimizer
non-deterministic even at fixed seed; the translational winner itself is
stable across all runs.) `plot_symmetry_type.py` renders this table as a
standalone publication figure
(`output_concrete_dimensionless/symmetry_type.png` / `.pdf`, 300 dpi):
the green translational bar against the two orange runners-up.

### 5.3 Generators

With `n = 8` dimensionless features and `k = 4` latent directions, there
are `n − k = 4` translational generators (components with `|g_j| > 0.05`
shown). Because the three binder fractions are collinear, **one of the
four is the trivial binder-simplex direction** (all binder fractions
moving together — a redundancy artifact, not a physical substitution);
the other three are strength-preserving mix substitutions:

| Generator | Dominant components | Physical reading |
|---|---|---|
| `g₁` | SP/b (+0.88), FlyAsh/b (−0.33), w/b (−0.27), Slag/b (+0.19) | Add superplasticizer while cutting fly ash and w/b |
| `g₂` | CoarseAgg/b (+0.83), ln(t/28) (+0.34), FlyAsh/b (−0.26), Cement/b (−0.25), w/b (−0.19), Slag/b (−0.17) | Add coarse aggregate and cure longer while trimming fly ash and cement |
| `g₃` | FineAgg/b (+0.91), FlyAsh/b (−0.35), Cement/b (−0.19), SP/b (−0.10) | Add fine aggregate while cutting fly ash |
| `g₄` *(trivial)* | Slag/b (+0.76), Cement/b (+0.53), CoarseAgg/b (+0.33) | Binder fractions rise together — the `m_c/b + m_s/b + m_f/b = 1` redundancy, not a physical direction |

The four vectors span the translational null space; because any
orthonormal basis of that 4-D space is equally valid, the individual
`gᵢ` directions (and their component signs) rotate from run to run — it is
the *subspace* they span, and the flatness of the model along it, that is
stable. The trivial binder-simplex direction always appears (it is forced
by the collinearity), leaving three physical strength-preserving
directions. Dropping one binder fraction (a 7-feature set) removes it and
yields three generators directly (see git history).

Each physical generator is a constant-residual direction in mix-ratio
space: moving the composition along `g_i` (within physical limits) leaves
the predicted strength residual `σ_c/σ_ideal` unchanged.

### 5.4 Figure

`output_concrete_dimensionless/concrete_symmetry_dimensionless.png`
reports: (left) measured strength against the Yeh baseline `σ_ideal`
with the 1:1 line (R² = 0.682); (centre) the learned latent embedding
coloured by the strength residual; (right) the validation-MSE bar chart
of the three competing symmetry candidates.

## 6. Validation of the Generators

The publication figure is produced by `plot_generator_lines.py`
(`output_concrete_dimensionless/generator_lines.png` / `.pdf`, 300 dpi):
take two real mixes from the dataset (a weaker and a stronger one), step
each one along all four generators, `π(ε) = π₀ + ε·g`, and feed every
synthetic recipe to the trained model. The result is **flat lines** —
the model-predicted `σc* = σc/σ_ideal` moves by ~1–2×10⁻⁷ (numerical
zero) as the recipe is changed along any generator (the trivial
binder-simplex generator is flat too — the model never depended on the
redundant fraction). For contrast, each mix is also stepped along the
model's strength-relevant direction (dashed): that line bends across the
full weak-to-strong span.

For the translational encoder this flatness is exact by construction:
the model computes `f(W·π)`, and each generator satisfies `W·g = 0`, so
`f(W·(π + ε·g)) = f(W·π)` for every ε. The figure therefore confirms the
trained model faithfully embodies the extracted generators — a
consistency check that the discovered symmetry directions are genuine
invariances of the fitted strength surface.

Suggested caption:

> **Figure X. Discovered strength-preserving directions in concrete mix
> design.** Two real mixes from the UCI dataset (a weaker and a stronger
> one) are stepped along each of the four Lie-algebra generators
> identified by the translational symmetry pipeline, `π(ε) = π₀ + ε·g`,
> and every synthetic recipe is fed to the trained model. Solid lines:
> the model-predicted dimensionless strength `σc* = σc/σideal` is held
> flat (change ~10⁻⁷) along every generator, where
> σideal = 13.83·(w/b)^(−1.269)·(0.268·ln t + 0.136) MPa is the
> regression baseline of Yeh (1998), evaluated with the literal
> water/binder ratio w/b = m_w/b. Dashed lines: stepping the same mixes
> along the model's most strength-relevant direction changes `σc*`
> across the full weak-to-strong span. For a translational encoder the
> flatness is exact by construction (`W·g = 0`), confirming the model
> embodies the discovered generators.

## 7. Reproducibility

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

The script defaults to `--encoder-hidden 64 32`, `raw_input=True`, and
`--latent-dim 4` (the latent dimension is pinned because the per-`k`
MSEs are nearly tied; pass `--latent-dim 0` to let the pipeline pick
`k` automatically), so no extra flags are required. Then produce the
publication figures (Sections 5.2 and 6):

```bash
python plot_generator_lines.py
python plot_symmetry_type.py
```

(`run_generator_check.py` runs all three steps in sequence and tees the
full transcript to `generator_check_full.log`.)

Output is written to `output_concrete_dimensionless/`:

- `concrete_symmetry_dimensionless.png` — three-panel summary figure.
- `run.log` — full console transcript (config, baseline fit, per-`k`
  metrics, symmetry losses, generator decomposition).
- `pipeline_artifacts.npz` — features, targets, encoder weights,
  generators, and symmetry losses of the run of record (input to the
  publication figures).
- `generator_lines.png` / `generator_lines.pdf` — publication figure:
  model-predicted `σc*` held flat along each generator vs bending along
  the strength direction (Section 6).
- `symmetry_type.png` / `symmetry_type.pdf` — publication figure:
  validation-MSE bar chart of the three competing symmetry families
  (Section 5.2).

## 8. Discussion

Non-dimensionalization factorizes the problem into a citable analytic
baseline and a learned dimensionless correction. The binder-referenced
ratios quotient out the overall "scale the whole mix" direction
analytically, and the Yeh baseline removes the two dominant physical
effects (w/b and age), so the network's entire capacity is spent on the
residual chemistry. The translational fingerprint recovered on these
coordinates confirms that the residual strength surface is governed by
additive combinations of the mix ratios. Of the four null-space
directions, one is the trivial binder-simplex redundancy (forced by the
three binder fractions summing to one); the remaining three provide an
interpretable, data-driven catalogue of strength-preserving mix
substitutions — e.g. adding superplasticizer while cutting fly ash and
w/b (`g₁`), or swapping fly ash for fine aggregate (`g₃`) — that can guide
constrained mix-design optimisation at a fixed target strength. The
publication figure (Section&nbsp;6) confirms the trained model embodies
these directions exactly: model-predicted `σc*` is held flat along every
generator (`W·g = 0`), while the strength-relevant direction spans the
full weak-to-strong range. The individual generator directions rotate
between retrainings, so the robust, reproducible claims are the
translational symmetry type and the three-dimensional *physical*
strength-preserving subspace — not any single named substitution.

## 9. References

1. I-C. Yeh, "Modeling of strength of high-performance concrete using
   artificial neural networks," *Cement and Concrete Research*,
   **28**(12), 1797–1808, 1998.
2. UCI Machine Learning Repository, Concrete Compressive Strength
   dataset #165. <https://archive.ics.uci.edu/dataset/165>
3. PyDimension Stage&nbsp;1: symmetry-aware dimensional-analysis pipeline
   (this repository, `projects/20260912_Stage1_Prokash/`).
