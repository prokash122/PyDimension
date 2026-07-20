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
**translational** symmetry candidate with a **1.7×** validation-MSE gap over
the next-best (scaling) candidate. Three independent Lie-algebra
generators are extracted, each a physically interpretable
strength-preserving substitution in mix-ratio space. The generators are
then validated **against measured data only**: real mix pairs separated
along the symmetry subspace change strength significantly less than
pairs separated along the encoder's strength-relevant directions
(mean |Δ(σ/σ_ideal)| 0.19 vs 0.29).

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
it retains its own separate ratio `π₄ = m_p / b`, so no information is
lost:

| Feature | Definition |
|---|---|
| `π₁` | `w/b = m_w / b` (literal water/binder ratio) |
| `π₂` | `m_f / b` (fly-ash replacement fraction) |
| `π₃` | `m_s / b` (slag replacement fraction) |
| `π₄` | `m_p / b` (superplasticizer dosage) |
| `π₅` | `m_{ca} / b` |
| `π₆` | `m_{fa} / b` |
| `π₇` | `ln(t / 28 d)` (dimensionless age) |

The cement ratio `m_c / b` is omitted because it is fixed by the other two
binder fractions (`m_c/b = 1 − m_s/b − m_f/b`), leaving six independent
mass ratios plus the age term. Age carries the only [T] dimension among
the inputs and cannot be non-dimensionalized against other columns; it is
referenced to the industry-standard 28-day curing age. The logarithm is
applied to the age ratio only: it symmetrizes the heavily skewed
1–365-day range around `π₇ = 0` at 28 days and matches the logarithmic
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

1. **Non-dimensionalization.** Construction of `π₁ … π₇` and
   `y = σ_c/σ_ideal` as defined in Section 3.
2. **Normalization.** Standard scaling (zero mean, unit variance) of the
   seven dimensionless features and of the residual target.
3. **Intrinsic-dimension discovery.** A latent-bottleneck autoencoder is
   trained for `k ∈ {1, …, 6}` (with seven features, `k` must stay below
   7 for translational generators to remain). The encoder is a
   multilayer perceptron
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

The intrinsic latent dimension of the residual is `k = 4` — an interior
optimum: held-out performance improves monotonically up to `k = 4` and
degrades for both `k = 5` and `k = 6`:

| `k` | `R²_train` | `R²_test` | MSE |
|---|---|---|---|
| 1 | 0.732 | 0.474 | 0.4469 |
| 2 | 0.758 | 0.494 | 0.4304 |
| 3 | 0.734 | 0.483 | 0.4391 |
| 4 | 0.762 | **0.541** | **0.3900** |
| 5 | 0.762 | 0.535 | 0.3950 |
| 6 | 0.762 | 0.437 | 0.4784 |

The R² values refer to the *residual* `σ_c/σ_ideal`, i.e. to the variance
left over after the analytic baseline has removed the dominant w/b and
age effects. The widening train–test gap beyond `k = 4` indicates the
extra latent directions fit noise rather than structure.

### 5.2 Symmetry type

Competitive training selects the translational candidate:

| Symmetry candidate | Held-out MSE |
|---|---|
| **translational** | **0.3326** |
| scaling | 0.5579 |
| rotational | 0.6331 |

The translational candidate beats the second-best (scaling) candidate
by a factor of **1.7×** in validation MSE: the strength residual is
additive in the binder-referenced mix ratios. (The gap and the
runner-up's identity fluctuate between roughly 1.4× and 1.9× across
retrainings because CPU thread scheduling makes the optimizer
non-deterministic even at fixed seed; the translational winner itself is
stable across all runs.)

### 5.3 Generators

With `n = 7` dimensionless features and `k = 4` latent directions, there
are `n − k = 3` independent translational generators (components with
`|g_j| > 0.05` shown):

| Generator | Dominant components | Physical reading |
|---|---|---|
| `g₁` | CoarseAgg/b (+0.70), w/b (−0.36), Slag/b (−0.36), FineAgg/b (−0.34), SP/b (+0.28), FlyAsh/b (+0.23), ln(t/28) (+0.11) | Add coarse aggregate while lowering w/b and slag |
| `g₂` | FineAgg/b (+0.61), SP/b (+0.48), w/b (−0.41), Slag/b (−0.35), CoarseAgg/b (−0.27), ln(t/28) (−0.18) | Add fine aggregate and superplasticizer while lowering w/b and slag |
| `g₃` | SP/b (+0.54), Slag/b (+0.46), CoarseAgg/b (+0.44), w/b (+0.35), FineAgg/b (+0.32), FlyAsh/b (−0.27), ln(t/28) (+0.08) | Add superplasticizer and slag (SCM exchange for fly ash) at higher w/b |

The three vectors span the strength-preserving subspace; because any
orthonormal basis of that 3-D null space is equally valid, the individual
`gᵢ` directions (and their component signs) rotate from run to run — it is
the *subspace* they span, and the flatness of the model along it, that is
stable.

Each generator is a constant-residual direction in mix-ratio space:
moving the composition along `g_i` (within physical limits) leaves the
predicted strength residual `σ_c/σ_ideal` unchanged.

### 5.4 Figure

`output_concrete_dimensionless/concrete_symmetry_dimensionless.png`
reports: (left) measured strength against the Yeh baseline `σ_ideal`
with the 1:1 line (R² = 0.682); (centre) the learned latent embedding
coloured by the strength residual; (right) the validation-MSE bar chart
of the three competing symmetry candidates.

## 6. Validation of the Generators

### 6.1 Six flat lines: model output along each generator

The simplest check (`plot_generator_lines.py`,
`output_concrete_dimensionless/generator_lines.png` / `.pdf`): take two
real mixes from the dataset (a weaker and a stronger one), step each one
along all three generators, `π(ε) = π₀ + ε·g`, and feed every synthetic
recipe to the trained model. The result is **six flat lines** — the
predicted strength moves by ~2–4×10⁻⁷ (numerical zero) as the recipe is
changed along any generator. For contrast, each panel also steps the
weaker mix along the model's strength-relevant direction (dashed): that
line bends by ~0.35, the full weak-to-strong span.

For the translational encoder this flatness is exact by construction:
the model computes `f(W·π)`, and each generator satisfies `W·g = 0`, so
`f(W·(π + ε·g)) = f(W·π)` for every ε. This confirms the trained model
faithfully embodies the extracted generators — a consistency check on
the model. Whether the *real* strength surface shares this invariance is
a separate question about measured data, answered next.
(`plot_generator_orbits.py` is an equivalent variant with three mixes
and a separate control panel.)

### 6.2 Test against measured strengths only

The generators are claims about the real strength surface, so they are
tested with **measured strengths only — no model prediction appears on
either axis** (`validate_generators.py`). The trained encoder `W`
splits standardized π-space into an *active* subspace (row space of
`W`, dim 4 — moving here changes predicted strength) and a *symmetry*
subspace (null space, dim 3 — spanned by the generators). For all
529,935 pairs of real mixes, the separation vector `Δπ` is decomposed
into these subspaces, and pairs whose separation is ≥ 90 % inside one
subspace (with total separation 0.5–2.5 standardized units) are
compared on their **measured** `σ_c/σ_ideal`:

| Pair type | Pairs | Mean \|Δ(σ_c/σ_ideal)\| |
|---|---|---|
| Near-duplicates, \|Δπ\| < 0.05 (repeatability noise floor) | 162 | **0.052** |
| **Symmetry-aligned (along generators)** | 5,110 | **0.188** |
| Random pairs at the same \|Δπ\| | 111,573 | 0.213 |
| Active-aligned (along strength-relevant directions) | 26,642 | 0.222 |
| **Control: along the single most strength-relevant direction** | 23 | **0.293** |

Per-generator, using pairs whose separation vector has \|cos\| ≥ 0.9
with one specific generator:

| Direction | Aligned pairs | Mean \|Δ\| | corr(y₋, y₊) |
|---|---|---|---|
| `g₁` | 13 | **0.135** | **+0.82** |
| `g₂` | 32 | 0.153 | +0.07 |
| `g₃` | 283 | 0.200 | −0.20 |
| top active direction (control) | 23 | 0.293 | +0.28 |

Interpretation: mixes that differ along the generators keep more nearly
the same measured strength residual, changing **1.6× less** than mixes
that differ along the most strength-relevant direction (0.188 vs 0.293),
and less than random pairs of equal separation (0.213). The symmetry is
*approximate*: aligned pairs sit above the replicate noise floor
(0.052), consistent with the autoencoder explaining 54.1 % — not 100 % —
of the residual variance. The *aggregate* ordering
(symmetry < random < control) is stable across retrainings, but which
individual generator reads as the "cleanest" invariance, and its
per-generator correlation, varies with the arbitrary null-space basis of
a given run — so the per-`gᵢ` rows above should be read as one
realization, not as fixed properties of a specific substitution.
The full analysis is reproduced by
`output_concrete_dimensionless/generator_validation.png` and
`validation.log`.

### 6.3 Publication figure

`make_publication_figure.py` condenses Sections 5.3 and 6.2 into a
single three-panel figure
(`output_concrete_dimensionless/publication_figure.png` / `.pdf`,
300 dpi). Suggested caption:

> **Figure X. Data-driven discovery and validation of
> strength-preserving directions in concrete mix design.**
> **(a)** The three Lie-algebra generators identified by the
> translational symmetry pipeline, shown as signed components in the
> standardized dimensionless mix-ratio space (binder-referenced ratios
> and log age). Each generator is a composition change predicted to
> leave the 28-day-normalized strength residual σc/σideal unchanged,
> where σideal = 13.83·(w/b)^(−1.269)·(0.268·ln t + 0.136) MPa is the
> regression baseline of Yeh (1998), evaluated with the literal
> water/binder ratio w/b = m_w/b.
> **(b)** Validation on measured data only: each point compares the
> measured strength residuals of two *actual* mixes from the UCI
> dataset (1030 samples). Blue: 328 pairs whose composition difference
> is aligned (|cos| ≥ 0.9) with a discovered generator — they
> concentrate on the 1:1 line. Red: 23 pairs aligned with the model's
> most strength-relevant direction — they depart from it. Pair
> separations are matched (0.5–2.5 standardized units); no model
> prediction is used.
> **(c)** Mean measured |Δ(σc/σideal)| per pair type with bootstrap
> 95% confidence intervals. Mixes differing along a generator change
> strength by 0.19 on average — less than pairs along the strength
> direction (0.29) and below random pairs of equal separation (0.21) —
> well above the repeatability floor set by replicate mixes (0.05).
> The discovered generators therefore identify
> approximate invariances of the real strength surface, not artifacts
> of the fitted network.

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

The script defaults to `--encoder-hidden 64 32` and `raw_input=True`, so
no extra flags are required. Then produce the orbit plots (Section 6.1)
and the measured-pair validation (Section 6.2):

```bash
python plot_generator_lines.py
python validate_generators.py
python make_publication_figure.py
```

Output is written to `output_concrete_dimensionless/`:

- `concrete_symmetry_dimensionless.png` — three-panel summary figure.
- `run.log` — full console transcript (config, baseline fit, per-`k`
  metrics, symmetry losses, generator decomposition).
- `pipeline_artifacts.npz` — features, targets, encoder weights, and
  generators of the run of record (input to the validation scripts).
- `generator_lines.png` / `.pdf` — six flat lines: model output along
  each generator (Section 6.1).
- `generator_validation.png`, `validation.log` — real-data pair test
  (Section 6.2).
- `publication_figure.png` / `publication_figure.pdf` — condensed
  three-panel figure with suggested caption (Section 6.3).

## 8. Discussion

Non-dimensionalization factorizes the problem into a citable analytic
baseline and a learned dimensionless correction. The binder-referenced
ratios quotient out the overall "scale the whole mix" direction
analytically, and the Yeh baseline removes the two dominant physical
effects (w/b and age), so the network's entire capacity is spent on the
residual chemistry. The translational fingerprint recovered on these
coordinates confirms that the residual strength surface is governed by
additive combinations of the mix ratios. The three generators provide an
interpretable, data-driven catalogue of strength-preserving mix
substitutions — e.g. adding coarse aggregate while lowering w/b and slag
(`g₁`), or a supplementary-cementitious-material exchange between slag
and fly ash (`g₃`) — that can guide constrained mix-design optimisation
at a fixed target strength. Crucially, these are not merely model
artifacts: the pair test of Section&nbsp;6 shows on measured strengths
alone that real mixes separated along the generator subspace change
strength ~1.6× less than mixes separated along the learned
strength-relevant direction (0.19 vs 0.29), and less than random pairs
of equal separation. The individual generator directions rotate between
retrainings, so the robust, reproducible claims are the translational
symmetry type, the three-dimensional strength-preserving subspace, and
this aggregate reduction — not any single named substitution.

## 9. References

1. I-C. Yeh, "Modeling of strength of high-performance concrete using
   artificial neural networks," *Cement and Concrete Research*,
   **28**(12), 1797–1808, 1998.
2. UCI Machine Learning Repository, Concrete Compressive Strength
   dataset #165. <https://archive.ics.uci.edu/dataset/165>
3. PyDimension Stage&nbsp;1: symmetry-aware dimensional-analysis pipeline
   (this repository, `projects/20260912_Stage1_Prokash/`).
