# Methods: Symmetry-Aware Latent-Variable Discovery

This document is the **paper-ready Methods section** for the PyDimension
Stage&nbsp;1 pipeline. It generalizes the per-example READMEs in
`Examples/` and describes only the operations that are common to every
case study (concrete strength, LHC dijets, laser-keyhole formation, LPBF
porosity, porous-media LBM flow).

> Throughout, let `X ∈ ℝ^{N×n}` denote the matrix of `N` samples and `n`
> physical input variables, and `y ∈ ℝ^{N}` denote the scalar target.
> Sets of pre-computed dimensionless candidates Π ∈ ℝ^{N×m} are
> optionally supplied from a Buckingham-Π analysis upstream of Stage&nbsp;1
> (Stage&nbsp;0 in the broader project).

---

## 1. Problem setting

We assume the target `y` depends on the inputs through a low-dimensional
**latent coordinate** `z ∈ ℝ^{k}` (`k ≤ n`) that is itself a function of
a symmetry-specific feature map of `X`:

```
y  ≈  f( z ),         z = W · ϕ(X),         ϕ ∈ { id, square, log }
```

where `W ∈ ℝ^{k×n}` is a linear projection and `ϕ` is a fixed,
per-symmetry transform. The Stage&nbsp;1 task is, from `(X, y)` alone, to
recover:

1. the intrinsic dimension `k`,
2. the symmetry class of `ϕ`,
3. the projection `W`,
4. the Lie-algebra generators of the symmetry group fixing `y`.

---

## 2. Preprocessing

All variables are standardised with zero mean and unit variance:

```
X̃_{ij} = (X_{ij} − μ_j) / σ_j,        ỹ_i = (y_i − μ_y) / σ_y .
```

Standard scaling is preferred over min–max because the Step&nbsp;3 scaling
encoder operates on `log|X|`, and shifting raw `X` near zero would inject
spurious large-magnitude features after the log. Means and variances are
fit on the full dataset and stored for inverse transforms when plotting
in physical units.

When a dimensional analysis is performed upstream, the corresponding
dimensionless candidates `Π` are normalised by the same procedure and
either (i) injected alongside the raw augmentation, or (ii) used as the
sole encoder input in **Π-only mode** (Sec. 4.4).

The Π *basis* — the integer exponents `α_{ij}` such that
`Π_j = ∏_i x_i^{α_{ij}}` — is obtained from
`pydimension.data_preprocessing.DataPreprocessor.process_with_dimensional_analysis()`
in the keyhole, LPBF-porosity, and porous-media-LBM examples, with an
inline `scipy.linalg.null_space` + SymPy integer-simplification fallback
used only when the repository pipeline cannot be loaded. The per-sample
numerical values of the Π features are then evaluated in-script as
`Π_j(x) = ∏_i x_i^{α_{ij}}` (or, where the physics dictates a specific
parametrisation, as the dedicated combination such as
`log_{10} Re` and `φ` for the porous-media case). The LHC dijet example
does *not* use `DataPreprocessor` for its Π candidates: the
dimensionless quantities of interest (`cos Δφ`, signed `p_T` ratios,
opening angles) are not power-law products of the raw momentum
components and are constructed inline. The concrete-strength example
does not use Π features at all, since its additive ansatz operates
directly on the raw mass densities.

---

## 3. Intrinsic-coordinate discovery (Step 2)

### 3.1 Architecture

For each candidate latent dimension `k ∈ {1, …, k_max}` (with
`k_max = 4` by default), a **bottleneck autoencoder** is trained:

```
        Encoder E_k                    Decoder D_k
   ϕ(X) ──► h₁ ──► h₂ ──► z ∈ ℝ^k ──► h̃₁ ──► h̃₂ ──► ŷ ∈ ℝ
        [64]    [32]                    [64]    [64]
```

- **Encoder.** A multilayer perceptron (MLP) with default hidden widths
  `[64, 32]` and `Tanh` activations followed by a linear projection to
  `ℝ^k`. Every example in the repository now uses this multilayer
  default; a single-layer linear encoder is available only by passing
  `encoder_hidden_dims=None` for ablation.
- **Decoder.** A two-hidden-layer MLP (`64 → 64`) with `Tanh`
  activations and a scalar output.
- **Joint training.** Encoder and decoder are trained jointly to
  minimise `MSE(D_k(E_k(ϕ(X))), y)` with Adam (`lr = 10⁻³`), batch size
  256, for `n_epochs = 600`. The optimiser uses a cosine learning-rate
  schedule decaying to `lr/100`.

### 3.2 Feature augmentation

In the default ("non-Π-only") mode, the encoder consumes the
**triple-feature augmentation**

```
ϕ_aug(X) = [ X,   X²,   log(|X|).clamp_min(0.1) ] ∈ ℝ^{N × 3n}
```

so a single linear encoder layer is expressive enough to represent any
of the three candidate symmetries (translational `X`, rotational `X²`,
scaling `log|X|`) and the joint MLP can further compose them. When
dimensionless candidates `Π` are provided they are concatenated to the
augmentation, yielding an `3n + m` input vector. In **Π-only mode** the
augmentation is bypassed (`raw_input=True`) and the encoder consumes the
normalised `Π` features directly — this is the mode used for the
keyhole, LPBF-porosity and porous-media examples.

### 3.3 Model selection

For every `k` we record both training and held-out (20 %) split metrics
over `n_restarts = 3` random seeds. Two quantities are reported:

```
R²_test(k) = 1 − Σ(ŷ_i − y_i)² / Σ(y_i − ȳ)²,        MSE_test(k).
```

The intrinsic dimension is selected as

```
k* = argmin_k  MSE_test(k),
```

subject to the side condition `R²_test(k*) ≥ 0.95 · max_k R²_test(k)`,
which prevents promoting an over-parameterised `k` that improves MSE
only marginally. We additionally inspect the `R²_test(k)` curve for an
"elbow" — the smallest `k` at which the curve plateaus — and report any
disagreement with the MSE-optimal `k*`.

---

## 4. Symmetry-type identification (Step 3)

### 4.1 Candidate symmetries

Three symmetry classes are tested:

| class | feature map `ϕ_s` | physical interpretation |
|---|---|---|
| translational | `X` | additive substitution: `y(x + ε g) = y(x)` for `W g = 0` |
| rotational    | `X²` (component-wise) | quadratic invariants: `y(R x) = y(x)` for `R ∈ SO(n)` mixing equal-weight inputs |
| scaling       | `log\|X\|.clamp_min(0.1)` | multiplicative invariance: `y(x ⊙ exp(ε s)) = y(x)` for `W s = 0` |

For every class `s ∈ {trans, rot, scal}` a **single-layer linear**
encoder

```
E_s(X) = W_s · ϕ_s(X),     W_s ∈ ℝ^{k* × n}
```

is paired with a *freshly initialised* decoder of identical
architecture to Step&nbsp;3.2 and trained jointly to minimise the held-out
MSE. The single-layer choice is deliberate: Step&nbsp;3 is a *symmetry
classifier*, not a regressor, and a one-layer encoder cleanly separates
the contributions of `W_s` and `ϕ_s` so that the winning class can be
read off the validation loss.

### 4.2 Training protocol

For each symmetry class:

- `n_restarts = 3` random initialisations,
- `n_epochs = 1500` Adam steps with cosine LR decay,
- weight decay `10⁻⁴`,
- batch size 256,
- 20 % validation split (the same indices for every class so the losses
  are directly comparable).

Restarts are seeded per-class via
`seed + hash(class) mod 1000 + 37 · r`, ensuring reproducible but
class-decorrelated initialisations. For each class only the
best-restart loss and weight matrix are retained.

### 4.3 Class decision

Let `L_s` be the best held-out MSE achieved by class `s`. The detected
symmetry class is

```
s* = argmin_s  L_s,
```

and the **loss gap** is reported as

```
gap = L_{second-best} / L_{s*} .
```

A gap of `> 3` is treated as a confident detection. When the gap is
below this threshold we either (i) revisit feature preprocessing (e.g.
log centring before the scaling encoder), or (ii) report a tied
detection.

### 4.4 Pi-only mode

For problems with a known Buckingham-Π reduction, Step&nbsp;2 can be run
in **Π-only mode** where the autoencoder consumes only the
dimensionless candidates. Step&nbsp;3, however, *always* runs on the raw
physical `X` (after standard scaling). The reason is physical: the
generators we ultimately want to extract live in physical-variable
space, and the per-class feature maps `ϕ_s` are defined on `X`, not on
`Π`. The scaling encoder in particular requires positive raw inputs so
that `log|X|.clamp_min(0.1)` reduces to a clean `log X` over the
sample range — this is why we use standard scaling rather than
min–max in Step&nbsp;1.

---

## 5. Generator extraction (Step 4)

Given the winning class `s*` and its linear weight matrix
`W ∈ ℝ^{k* × n}`, the Lie-algebra generators of the symmetry group
fixing `y` are obtained in closed form:

### 5.1 Translational and scaling

The generators are the **null-space basis** of `W`:

```
{ g ∈ ℝ^n : W g = 0 } = span{ g_1, …, g_{n−k*} } ,
```

computed via SVD with `scipy.linalg.null_space`. Each `g_i` is a
direction in input space along which the encoded coordinate is
invariant:

- **translational:** `x → x + ε g_i` keeps `z = W x` (and hence `y`)
  unchanged — this is the "mix substitution" reading used for the
  concrete example.
- **scaling:** the same null-space vector `s_i` is exponentiated:
  `x → x ⊙ exp(ε s_i)`, i.e. a power-law rescaling of subsets of inputs
  by reciprocal exponents — used in the porous-media LBM Reynolds-number
  collapse.

### 5.2 Rotational

For the rotational class, the encoder weight row gives a vector
`w ∈ ℝ^n` of quadratic coefficients with `z = Σ_i w_i x_i²`. Indices
with **equal `|w_i|`** can be mixed by an orthogonal rotation without
changing `z`. We cluster indices by absolute weight using a relative
tolerance (`cluster_tol = 0.25`) and emit one antisymmetric generator

```
A_{(ij)} = e_i e_j^T − e_j e_i^T,        x → exp(ε A_{(ij)}) · x
```

for each pair `(i, j)` of indices within a cluster of size ≥ 2.

### 5.3 Verification

For every extracted generator we numerically verify the invariance by
propagating an orbit

```
x(τ) = x₀ + τ g            (translational)
x(τ) = x₀ ⊙ exp(τ s)       (scaling)
x(τ) = exp(τ A) · x₀       (rotational)
```

through the trained decoder and confirming that `|D(E(ϕ(x(τ)))) − D(E(ϕ(x₀)))|`
remains below a small tolerance over a finite `τ ∈ [−τ_max, τ_max]`
range. In every reported case the residual stays at the noise level of
the held-out fit.

---

## 6. Implementation summary

| Component | Default | Tunable knobs |
|---|---|---|
| Scaler | standard (zero mean, unit variance) | `method ∈ {standard, robust, minmax}` |
| Step 2 encoder | MLP `[64, 32]` + linear projection, `Tanh` | `encoder_hidden_dims` |
| Step 2 decoder | MLP `[64, 64]` + linear, `Tanh` | `hidden_dim` |
| Step 2 training | Adam, `lr=1e-3`, 600 epochs, cosine LR | `n_epochs`, `lr`, `batch_size` |
| Step 2 restarts | 3 seeds, 20 % held out | `n_restarts`, `val_fraction` |
| Step 2 features | `[X, X², log|X|]` (+ Π) by default | `raw_input`, `pi_features` |
| Step 3 encoder | single linear layer per class, no bias | — |
| Step 3 training | Adam, `lr=1e-3`, weight decay `1e-4`, 1500 epochs | same knobs as Step 2 |
| Step 3 classes | translational / rotational / scaling | — |
| Step 4 (trans./scal.) | SVD null space of `W` | — |
| Step 4 (rot.) | clustered antisymmetric pairs | `cluster_tol` |

All randomness is controlled by a single integer seed (`seed = 42` in
every example), and the canonical reproduction command for an example
directory is:

```bash
python discover_symmetry.py \
    --data <input file> \
    --encoder-hidden 64 32 \
    --seed 42 \
    --latent-epochs 600 \
    --sym-epochs 1500 \
    --n-restarts 3
```

Console transcripts are saved to `output_<name>/run.log` and summary
figures to `output_<name>/<name>_symmetry_discovery.png`.

---

## 7. Differences between examples

The pipeline above is shared by all five case studies. The
per-example knobs differ only in:

| Example | Step 2 features | Step 2 encoder | Notes |
|---|---|---|---|
| Concrete compressive strength | `[X, X², log\|X\|]` (no Π) | MLP `[64, 32]` | Pure additive (translational) discovery |
| LHC dijets | raw 4-vector + 6 Π candidates (e.g. `cos Δφ`, `p_T` ratios) | MLP `[64, 32]` | Rotational SO(2) on `(p_{1x}, p_{1y}, p_{2x}, p_{2y})` |
| Laser keyhole | `[X, X², log\|X\|]` + reduced Π set | MLP `[64, 32]` | Π-only by default; scaling symmetry |
| LPBF porosity | `[X, X², log\|X\|]` + reduced Π set | MLP `[64, 32]` | Π-only by default; scaling symmetry |
| Porous-media LBM | Π features only (`raw_input=True`) | MLP `[64, 32]` | Re-number collapse via scaling generators |

In every case, **Step 3 runs on the raw physical `X` (standard-scaled
only)** so that the per-class feature maps `ϕ_s` operate in their
intended space and the extracted generators are directly interpretable
in physical units.

---

## 8. Reporting checklist

For each application we report:

1. The dataset (size, variables, units, source).
2. The Stage&nbsp;2 metric curve `{(k, R²_test(k), MSE_test(k))}` and the
   selected `k*`.
3. The Stage&nbsp;3 loss table `{L_trans, L_rot, L_scal}` and the loss gap.
4. The Stage&nbsp;3 weight matrix `W` (or its dominant row for `k* = 1`).
5. The Stage&nbsp;4 generators with their physical interpretation.
6. A summary figure with at least: (i) latent embedding vs target,
   (ii) symmetry-class MSE bar chart.

This checklist matches the structure used in each per-example README
and ensures one-to-one mapping between the textual claims in the paper
and the artefacts emitted by the code.
