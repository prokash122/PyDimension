# Regime-Aware Symmetry Discovery — Mathematical Formulation

This note states, in mathematical form:

1. **Their method** — *clustering dimensionless learning* (Zhang, Xu,
   Wang & He, CMAME 420 (2024) 116728);
2. **Our method** — the PyDimension Stage-1 symmetry-discovery +
   encoder-L2 equation extraction used elsewhere in this example;
3. **The linkage** — why the two are mathematically compatible (they
   optimise the *same object* in the *same space*, one locally and one
   globally);
4. **The combined algorithm** implemented in
   `discover_regimes_clustering.py`.

---

## 0. Common setting and notation

A physical system has $m$ independent dimensional quantities
$\mathbf{q} = [q_1,\dots,q_m]^T$ with $l$ fundamental dimensions and a
dimensionless output $\pi$. The dimension matrix
$\mathbf{D}\in\mathbb{Z}^{l\times m}$ collects the dimension exponents.
By the Buckingham-Pi theorem, the null space of $\mathbf{D}$ has
dimension $n = m - l$ and yields independent dimensionless groups

$$
\pi_i = \prod_{j=1}^m q_j^{e_{i,j}},\qquad \mathbf{D}\mathbf{e}_i = \mathbf{0},
\qquad i = 1,\dots,n,
$$

with the physical relationship

$$
\pi = f(\pi_1,\dots,\pi_n) = g(x_1,\dots,x_n),\qquad x_i = \log \pi_i .
$$

The vector $\mathbf{x} = \log\boldsymbol{\Pi}$ lives in **log-Pi
space**; any product of powers of the $\pi_i$ is a *linear* functional
$\mathbf{w}\cdot\mathbf{x}$ of $\mathbf{x}$. Both methods below operate
in this space; both seek the *directions* $\mathbf{w}$ along which $g$
actually varies.

**This case.** For porous-media flow the inputs are
$(\rho, v, d, \mu, \varphi)$ with output the friction factor
$f = dP_L\, d/(\rho v^2)$. Dimensional analysis gives
$\Pi = (Re_p, \varphi)$ with $Re_p = \rho v d/\mu$ (the solid fraction
$1-\varphi$ is carried as an explicit third coordinate in the equation
extraction, cf. the main README). Ground truth (never shown to any
algorithm) is the Ergun equation

$$
f \;=\; \Big[\underbrace{\tfrac{150\,(1-\varphi)}{Re_p}}_{\text{viscous}}
\;+\; \underbrace{1.75}_{\text{inertial}}\Big]\,
\frac{1-\varphi}{\varphi^{3}},
$$

a **sum of two power laws**: no single $\mathbf{w}$ works globally, but
each limit is an exact monomial —

$$
f_{\text{visc}} = 150\, Re_p^{-1}\varphi^{-3}(1-\varphi)^{2}
\quad\text{for } \tfrac{Re_p}{1-\varphi}\ll 85.7,
\qquad
f_{\text{iner}} = 1.75\, \varphi^{-3}(1-\varphi)
\quad\text{for } \tfrac{Re_p}{1-\varphi}\gg 85.7 .
$$

This is precisely the "multiple-physical-regime" situation the paper
addresses and the "hidden scaling symmetry per regime" situation
Stage-1 addresses.

---

## 1. Their method: clustering dimensionless learning

### 1.1 Active subspace (single regime)

Define the uncentred covariance of the gradient of $g$,

$$
\mathbf{C} \;=\; \int \nabla g(\mathbf{x})\, \nabla g(\mathbf{x})^T\,
\phi(\mathbf{x})\, d\mathbf{x}
\;=\; \mathbf{W}\boldsymbol{\Lambda}\mathbf{W}^T,
\qquad \lambda_1 \ge \cdots \ge \lambda_n \ge 0,
$$

with weight $\phi(\mathbf{x}) = 1/\lVert\nabla g\rVert^2$
(normalisation). Because

$$
\int \Big(\frac{\partial \pi}{\partial \hat{x}_i}\Big)^2 \phi\, d\mathbf{x}
= \mathbf{w}_i^T \mathbf{C}\, \mathbf{w}_i = \lambda_i,
\qquad \hat{x}_i = \mathbf{x}^T\mathbf{w}_i ,
$$

$\lambda_i$ measures the mean-square variation of $\pi$ along the
direction $\mathbf{w}_i$. If $\lambda_p \gg \lambda_{p+1}$ then

$$
\pi \approx \hat{g}(\hat{x}_1,\dots,\hat{x}_p),
\qquad
\hat{\pi}_i = \exp(\hat{x}_i) = \prod_j \pi_j^{\,w_{i,j}},
$$

i.e. the system is governed by $p$ *new* dimensionless groups
$\hat\pi_i$ — products of powers of the original ones, read off the top
eigenvectors.

In practice $g$ is unknown: a Gaussian-process regression is fit to the
sample set $\Omega = \{(\mathbf{x}^{(i)}, \pi^{(i)})\}_{i=1}^N$, gradients
are obtained **analytically from the GPR posterior-mean formula**

$$
\mu(\mathbf{x}) = \sum_i \alpha_i\, k(\mathbf{x}, \mathbf{x}^{(i)}),
\qquad
\nabla\mu(\mathbf{x}) = \sum_i \alpha_i\, \nabla_{\mathbf{x}}
k(\mathbf{x}, \mathbf{x}^{(i)}),
\qquad \boldsymbol{\alpha} = \mathbf{K}^{-1}\mathbf{y},
$$

and the integral is replaced by the Monte-Carlo sum (paper Eq. 25)

$$
\mathbf{C}^I = \frac{1}{|\Omega^I|}\sum_{\mathbf{x}^{(i)}\in\Omega^I}
\frac{\nabla g(\mathbf{x}^{(i)})}{\lVert \nabla g(\mathbf{x}^{(i)})\rVert}
\frac{\nabla g(\mathbf{x}^{(i)})^T}{\lVert \nabla g(\mathbf{x}^{(i)})\rVert}.
$$

### 1.2 Regime identification by gradient clustering

A regime is a subset of data over which the *gradient direction field*
is coherent. The paper partitions $\Omega$ into clusters
$\Omega^1,\dots,\Omega^K$ by maximising

$$
S(\Omega^1,\dots,\Omega^K) = \sum_{I=1}^K \sum_{\mathbf{x}\in\Omega^I}
\mathrm{Sim}\big(\nabla g(\mathbf{x}), \Omega^I\big),
\qquad
\mathrm{Sim}\big(\nabla g, \Omega^I\big) =
\sum_{j=1}^n \frac{\lambda_j^I}{\lVert\boldsymbol{\lambda}^I\rVert}
\big(\hat{\mathbf{g}}\cdot \mathbf{w}_j^I\big)^2,
$$

where $\hat{\mathbf{g}} = \nabla g/\lVert\nabla g\rVert$ and
$(\boldsymbol{\lambda}^I, \mathbf{W}^I)$ are the eigenpairs of
$\mathbf{C}^I$. The K-means-style iteration alternates **similarity
matching** (assign each point to $\arg\max_I \mathrm{Sim}$) and
**eigenpair updates** (recompute $\mathbf{C}^I$). $K$ is selected by

- (a) $\lambda_1^I/\lambda_n^I \ge E$ for all $I$ (each cluster has a
  genuinely dominant direction; $E = 50$ suggested), and
- (b) $|\Omega^I|/|\Omega| \ge \epsilon$ (no vanishing clusters;
  $\epsilon = 5\%$).

Then the single-regime active-subspace analysis of §1.1 is run **per
cluster**.

---

## 2. Our method: Stage-1 symmetry discovery + encoder-L2 extraction

Stage-1 asks a complementary question: *what invariance group does the
data respect, and what monomial coordinate realises it?* It fits, by
gradient descent, a **global** latent model instead of analysing local
gradients.

### 2.1 Latent dimension ($k^*$)

An MLP encoder–regressor $f \approx \mathcal{D}(\mathcal{E}_k(\Pi))$ is
trained for $k = 1, 2, \dots$; the smallest $k$ reaching the $R^2$
plateau is the latent dimension $k^*$ — the number of coordinates the
output truly depends on.

### 2.2 Symmetry type

Three competing single-linear-layer encoders are trained jointly with a
shared decoder $F_\theta$ (Tanh MLP):

$$
z_{\text{scal}} = \mathbf{w}\cdot \log|\mathbf{X}|,\qquad
z_{\text{trans}} = \mathbf{w}\cdot \mathbf{X},\qquad
z_{\text{rot}} = \mathbf{w}\cdot \mathbf{X}^2 .
$$

The transform with the lowest reconstruction loss identifies the
invariance class. A win for $z_{\text{scal}}$ means the level sets of
$f$ are (locally) orbits of a **scaling group**: there exists
$\mathbf{w}$ with

$$
f(\Pi) \approx F\big(\mathbf{w}\cdot\log\Pi\big)
\;=\; F\big(\log \textstyle\prod_j \pi_j^{w_j}\big),
$$

i.e. $f$ is invariant under
$\pi_j \mapsto s^{\,u_j}\pi_j$ for every $\mathbf{u}\perp\mathbf{w}$
(the $n-k^*$ null directions of the encoder are the symmetry
generators).

### 2.3 Equation extraction from the L2 encoder (this example)

With $k^* = 1$ and scaling the winner, train the standalone pair
$(\mathbf{w}, F_\theta)$ by MSE with plain weight decay, then read the
law off the weight vector alone:

$$
\bar{\mathbf{w}} = \frac{\mathbf{w}}{\lVert\mathbf{w}\rVert},
\qquad
\log f = \alpha\,(\bar{\mathbf{w}}\cdot\mathbf{x}) + \log C
\;\;\text{(1-D OLS for } \alpha, C\text{)},
\qquad
\boxed{\,f = C\, \textstyle\prod_j \pi_j^{\alpha \bar{w}_j}\,}
$$

For this example $\mathbf{x} = [\log Re_p, \log\varphi,
\log(1-\varphi)]$ (mean-centred), so the discovered law reads
$f = C\, Re_p^{\alpha\bar w_1} \varphi^{\alpha\bar w_2}
(1-\varphi)^{\alpha\bar w_3}$.

**Limitation that motivates this work:** the model
$f \approx F(\mathbf{w}\cdot\mathbf{x})$ assumes **one** ridge direction
for the whole dataset. The Ergun equation is a sum of two monomials, so
run on mixed data Stage-1 has no valid global $\mathbf{w}$ — which is
why the main README's runs split the regimes *by hand* first.

---

## 3. The linkage

Both methods search for the same object — a dominant direction
$\mathbf{w}$ in log-Pi space — through two different lenses:

| | Their method (active subspace) | Our method (scaling encoder) |
|---|---|---|
| Object | top eigenvector $\mathbf{w}_1$ of $\mathbf{C}$ | encoder weight $\bar{\mathbf{w}}$ |
| Estimate | **local**: statistics of $\nabla g$ (GPR) | **global**: joint fit of $(\mathbf{w}, F_\theta)$ |
| Model class | ridge function $\pi \approx \hat g(\mathbf{x}^T\mathbf{w}_1,\dots)$ | ridge function $f \approx F(\mathbf{w}\cdot\mathbf{x})$ |
| Dominance test | spectral gap $\lambda_1 \gg \lambda_2$ | latent-dim sweep $k^* = 1$ |
| Output | dominant group $\hat\pi_1 = e^{\mathbf{x}\cdot\mathbf{w}_1}$ (direction only) | calibrated law $f = C\prod_j \pi_j^{\alpha\bar w_j}$ (direction **and** magnitude) |

The formal connection: **if** the data obey a single monomial law
$f = C\prod_j \pi_j^{a_j}$, then $g(\mathbf{x}) = \log C +
\mathbf{a}\cdot\mathbf{x}$ and

$$
\nabla g(\mathbf{x}) \equiv \mathbf{a}
\quad\Longrightarrow\quad
\mathbf{C} = \frac{\mathbf{a}\,\mathbf{a}^T}{\lVert\mathbf{a}\rVert^2},
\qquad
\lambda_1 = 1,\;\; \lambda_2 = \cdots = \lambda_n = 0,
\qquad
\mathbf{w}_1 = \frac{\mathbf{a}}{\lVert\mathbf{a}\rVert} = \bar{\mathbf{w}} .
$$

So in the exact-power-law limit the paper's top eigenvector and our
L2-normed encoder weight are **the same vector**, the spectral gap
$\lambda_1/\lambda_2 \to \infty$ certifies the same fact as
$k^* = 1$, and the paper's $\hat\pi_1$ is the argument of our $F$.
The methods diverge only in *how much data they trust at once*:

- the active subspace is built from **pointwise** gradients, so it
  degrades gracefully when the dataset contains several regimes — the
  gradients simply form several bundles, which is exactly what the
  paper's similarity clustering detects;
- the encoder is a **global** fit, so it fails on mixed data but, on
  single-regime data, delivers what the eigenvector cannot: the scale
  $\alpha$ and prefactor $C$, and (via the symmetry-type contest) the
  *certificate* that the invariance really is a scaling and not a
  translation or rotation.

Hence the natural division of labour: **their clustering finds the
regimes; our encoder turns each regime into an equation.** The
clustering replaces precisely one thing in this example's original
workflow — the manual construction of `dataset_ergun_viscous_widephi.csv`
and `dataset_ergun_inertial_widephi.csv` — and leaves the Stage-1
extraction untouched (same architecture, loss, optimiser and recipe as
`discover_equation_encoder_l2.py`).

---

## 4. The combined algorithm (`discover_regimes_clustering.py`)

Adaptations beyond the verbatim paper method are marked **[A1]–[A4]**
and justified below.

> **Input:** raw table $\{(\rho, v, d, \mu, \varphi, f)^{(i)}\}_{i=1}^N$
> spanning an unknown number of regimes. No regime labels.
> **Output:** number of regimes $K$, a regime label per row, a dominant
> dimensionless group per regime, and a calibrated power law per regime.
>
> **Step 1 — Dimensional analysis.**
> Buckingham-Pi: $\Pi = (Re_p, \varphi)$, $\mathbf{x} = \log\Pi$.
> Standardise per component: $\tilde{x}_d = (x_d - m_d)/s_d$ **[A1]**.
>
> **Step 2 — Regression.**
> Fit GPR $g(\tilde{\mathbf{x}}) \approx \ln f$ **[A2]** with kernel
> $\sigma^2\,\mathrm{RBF}_{\boldsymbol{\ell}} + \mathrm{White}$
> (anisotropic $\boldsymbol{\ell}$, hyperparameters by marginal
> likelihood).
>
> **Step 3 — Gradients.**
> Analytic posterior-mean gradients at every sample:
> $\displaystyle \nabla g(\tilde{\mathbf{x}}) = \sigma_y \sum_i \alpha_i\,
> k(\tilde{\mathbf{x}},\tilde{\mathbf{x}}^{(i)})\,
> \frac{\tilde{\mathbf{x}}^{(i)} - \tilde{\mathbf{x}}}{\boldsymbol{\ell}^2}$;
> normalise $\hat{\mathbf{g}} = \nabla g / \lVert\nabla g\rVert$.
> (Un-standardised, $\partial \ln f/\partial \ln \pi_d =$ local
> power-law exponents — reported as a diagnostic.)
>
> **Step 4 — Clustering (paper Eqs. 16–20).**
> For each $K$: from random partitions (20 restarts), iterate
> (i) $\mathbf{C}^I = \mathrm{mean}_{\,\Omega^I}\,
> \hat{\mathbf{g}}\hat{\mathbf{g}}^T
> = \mathbf{W}^I \boldsymbol{\Lambda}^I (\mathbf{W}^I)^T$;
> (ii) reassign each point to
> $\arg\max_I \sum_j (\lambda^I_j/\lVert\boldsymbol{\lambda}^I\rVert)
> (\hat{\mathbf{g}}\cdot\mathbf{w}^I_j)^2$
> — until fixed point; keep the restart maximising $S$.
>
> **Step 5 — Select $K$.**
> Evaluate criteria (a) $\lambda^I_1/\lambda^I_n \ge E$ and
> (b) $|\Omega^I|/N \ge \epsilon$ for $K = 1,\dots,K_{\max}$; report the
> table; analyse the main-regime level (smallest $K$ resolving distinct
> gradient bundles — here $K = 2$).
>
> **Step 6 — Per-regime active subspace (paper Eqs. 25–26).**
> For each cluster: eigenpairs of $\mathbf{C}^I$; dominant group
> $\hat\pi^I_1 = \exp(\mathbf{x}\cdot\mathbf{w}^I_1)$ with exponents
> de-standardised as $w_{1,d}/s_d$; spectral gap
> $\lambda^I_1/\lambda^I_2$ quantifies its dominance.
>
> **Step 7 — Per-regime Stage-1 extraction (ours).**
> For each cluster $\Omega^I$ (and for its **core**
> $\{\text{margin}_i \ge Q_{0.25}\}$, where
> $\text{margin}_i = \mathrm{Sim}_{\text{own}} -
> \max_{J\ne I}\mathrm{Sim}_J$ **[A3]**):
> train the scaling encoder + Tanh decoder
> $(\mathbf{w}, F_\theta)$ on
> $\mathbf{x} = [\log Re_p, \log\varphi, \log(1-\varphi)]$, 8 restarts;
> for each restart do the 1-D OLS
> $\log f = \alpha(\bar{\mathbf{w}}\cdot\mathbf{x}) + \log C$ and keep
> the restart with the highest $R^2(\log f)$ **[A4]**; report
> $f = C \prod_j \pi_j^{\alpha \bar w_j}$ per regime.

**Adaptations.**

- **[A1] Standardised logs.** $\log Re_p$ spans ~12 natural-log decades,
  $\log\varphi$ under 1; the paper's own Sec. 2.4 discussion prescribes
  normalising the logarithms when their ranges are incomparable
  (otherwise the similarity is blind to the narrow coordinate).
- **[A2] Regress $\ln f$, not $f$.** Here $f$ spans 10 orders of
  magnitude, making GPR on $f$ ill-conditioned; with $g = \ln f$ the
  gradients become local exponents — the natural relevance measure for
  scaling laws (same spirit as the paper's use of logarithmic
  derivatives $\partial\pi/\partial x_i$).
- **[A3] Core extraction.** The similarity margin is smallest at regime
  interfaces, so trimming the lowest-margin quartile removes
  transition-zone contamination *using only quantities the clustering
  already computes* — no ground truth.
- **[A4] Restart selection by power-law $R^2(\log f)$.** On the
  $\varphi/(1-\varphi)$ data manifold there is a one-parameter family of
  $(\mathbf{w}, F_\theta)$ with near-identical decoder MSE (see the main
  README's identifiability caveat), so decoder test-MSE cannot choose
  between equivalence-class members. The quantity being *claimed* is a
  monomial law; among equally-fitting encoders, the direction whose 1-D
  OLS is most consistent with a global power law is the one to report.
  The criterion is sign-invariant and uses no ground truth.

**Committed result** (`output_regime_aware/`): $K = 2$ reproduces the
analytic regime boundary $Re_p/(1-\varphi) = 150/1.75 \approx 85.7$
with 95.1 % agreement, and the per-regime extractions give

$$
f_{\text{visc}}^{\text{disc}} = 198.3\; Re_p^{-0.986}\,
\varphi^{-2.924}\,(1-\varphi)^{+2.150}
\;(R^2 = 0.993),
\qquad
f_{\text{iner}}^{\text{disc}} = 1.85\; Re_p^{-0.003}\,
\varphi^{-2.982}\,(1-\varphi)^{+1.019}
\;(R^2 = 0.997),
$$

against the true $150\,Re_p^{-1}\varphi^{-3}(1-\varphi)^2$ and
$1.75\,\varphi^{-3}(1-\varphi)$ — both limits of the Ergun equation
recovered from one mixed dataset with no manual regime split.
