# Data-Driven Discovery of Translational Symmetry in Concrete Compressive Strength

## Abstract

We apply the PyDimension Stage&nbsp;1 symmetry-discovery pipeline to the UCI
Concrete Compressive Strength dataset (Yeh, 1998; 1030 samples, 8 mix-design
inputs) to test whether the governing functional form is consistent with an
**additive (translational)** symmetry. A multilayer perceptron (MLP) encoder
with hidden widths `[64, 32]` is trained jointly with a paired decoder. The
intrinsic latent dimension is identified via held-out reconstruction
performance, and the symmetry type is determined by a competitive
encoder-training step over translational, rotational, and scaling
candidates. The translational candidate is selected with a 3.7&times;
validation-MSE gap over the next-best (scaling) candidate, and five
independent Lie-algebra generators are extracted. Each generator
corresponds to a physically interpretable **mix substitution** that
preserves compressive strength.

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
   with hidden widths `[64, 32]` and `Tanh` activations; the decoder is a
   paired MLP of matching capacity. Each `k` is repeated over `n_restarts = 3`
   random seeds and 600 epochs, and the latent dimension minimizing the
   held-out reconstruction MSE is selected.
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

The intrinsic latent dimension is `k = 3`. The held-out coefficient of
determination is `R² = 0.892` at `k = 3`, with a near-degenerate plateau
between `k = 1` and `k = 3`:

| `k` | `R²_train` | `R²_test` | MSE |
|---|---|---|---|
| 1 | 0.970 | 0.892 | 0.0999 |
| 2 | 0.970 | 0.862 | 0.1280 |
| 3 | 0.973 | **0.892** | **0.0996** |
| 4 | 0.972 | 0.881 | 0.1104 |

### 4.2 Symmetry type

Competitive training cleanly selects the translational candidate:

| Symmetry candidate | Held-out MSE |
|---|---|
| **translational** | **0.1223** |
| scaling | 0.4496 |
| rotational | 0.5789 |

The translational candidate beats the second-best (scaling) candidate by a
factor of **3.7&times;** in validation MSE, confirming that the governing
combination of inputs is additive rather than multiplicative.

### 4.3 Generators

With `n = 8` inputs and `k = 3` latent directions, there are
`n − k = 5` independent translational generators. The dominant components
of each are listed below (only `|g_j| > 0.05` shown):

| Generator | Dominant components | Physical reading |
|---|---|---|
| `g₁` | Water (+0.98), Fly Ash (+0.16), Slag (+0.09), Cement (+0.09) | Uniform water rescaling at fixed binder share |
| `g₂` | Superplast. (+0.70), Fly Ash (−0.60), Fine Agg. (−0.30), Slag (+0.23) | Replace fly ash + fine aggregate with superplasticizer + slag |
| `g₃` | Coarse Agg. (+0.99), Fly Ash (−0.13) | Replace fly ash with coarse aggregate |
| `g₄` | Fine Agg. (+0.71), Fly Ash (−0.62), Superplast. (−0.29), Slag (+0.14) | Replace fly ash + superplasticizer with fine aggregate + slag |
| `g₅` | Cement (+0.73), Slag (−0.58), Fly Ash (−0.29), Fine Agg. (−0.16) | Replace slag + fly ash with cement |

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
    --encoder-hidden 64 32 \
    --seed 42 \
    --latent-epochs 600 \
    --sym-epochs 1500 \
    --n-restarts 3
```

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
a three-dimensional latent manifold, while still preserving the
translational character of the symmetry. The five extracted generators
provide an interpretable, data-driven catalogue of strength-preserving
mix substitutions that can guide constrained mix-design optimization
(e.g. supplementary cementitious material substitution at fixed target
strength).

## 7. References

1. I-C. Yeh, "Modeling of strength of high-performance concrete using
   artificial neural networks," *Cement and Concrete Research*,
   **28**(12), 1797–1808, 1998.
2. UCI Machine Learning Repository, Concrete Compressive Strength
   dataset #165. <https://archive.ics.uci.edu/dataset/165>
3. PyDimension Stage&nbsp;1: symmetry-aware dimensional-analysis pipeline
   (this repository, `projects/20260912_Stage1_Prokash/`).
