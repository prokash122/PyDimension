"""
Discover the scaling symmetry in 3D porous media flow data from a Lattice
Boltzmann Method (LBM) simulator using the PyDimension Stage1 pipeline.

Physics
-------
3D flow through a random sphere-packed porous medium follows the Ergun
equation:

    f = [150·(1-φ)/Re_p + 1.75] · (1-φ)/φ³

where
    f    = (dP_L · d) / (rho · v²)     friction factor   (dimensionless OUTPUT)
    Re_p = rho · v · d / mu            particle Reynolds number
    phi  = porosity

The pipeline does NOT know this formula -- it recovers the dimensionless-group
structure, the latent dimension k* = 2, the scaling symmetry, and the
Lie-algebra generators directly from LBM data.

Variables (6 physical inputs)
-----------------------------
    | Variable                | Symbol  | Units    | Dimensions      |
    |-------------------------|---------|----------|-----------------|
    | Pressure gradient       | dP_L    | Pa/m     | kg·m⁻²·s⁻²      |
    | Superficial velocity    | v       | m/s      | m·s⁻¹           |
    | Dynamic viscosity       | mu      | Pa·s     | kg·m⁻¹·s⁻¹      |
    | Fluid density           | rho     | kg/m³    | kg·m⁻³          |
    | Particle diameter       | d       | m        | m               |
    | Porosity                | phi     | -        | dimensionless   |

5 dimensional inputs + 1 already-dimensionless (phi) = 6 columns total.
Three fundamental dimensions (M, L, T) → 5 - 3 = 2 Pi groups from dimensional
variables; plus phi → 3 dimensionless groups total:

    Pi_phi = phi
    Re_p   = rho · v · d / mu
    f      = (dP_L · d) / (rho · v²)            <-- the OUTPUT

Usage
-----
    python discover_symmetry.py --data dataset_lbm_porous.csv
"""

import sys
import os
import argparse
import traceback
import multiprocessing

import numpy as np
import torch

# Add the Stage1 project to the path
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_here, "..", ".."),
    os.path.join(_here, "..", "..", "projects", "20260912_Stage1_Prokash"),
    _here,
]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, "preprocessing")):
        sys.path.insert(0, _candidate)
        break

try:
    import matplotlib
    matplotlib.use("Agg")
except (AttributeError, ImportError):
    pass
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":             14,
    "axes.titlesize":        17,
    "axes.labelsize":        16,
    "xtick.labelsize":       14,
    "ytick.labelsize":       14,
    "legend.fontsize":       14,
    "legend.title_fontsize": 15,
    "figure.titlesize":      21,
})

try:
    from preprocessing.normalize import normalize_data
    from intrinsic_coordinate.discovery import discover_latent_dimension
    from symmetry_discovery.identification import identify_symmetry
    from symmetry_discovery.generators import extract_generators
except ImportError as e:
    print(f"ERROR: Could not import Stage1 modules: {e}")
    sys.exit(1)

# Required: the repository's DataPreprocessor for Buckingham-Pi reduction.
# Install pydimension (and its `seaborn` dependency) if the import below
# fails — this is the only Pi-discovery path the script supports.
_da_root = _here
while _da_root and not os.path.isdir(os.path.join(_da_root, "pydimension")):
    nxt = os.path.dirname(_da_root)
    if nxt == _da_root:
        break
    _da_root = nxt
if os.path.isdir(os.path.join(_da_root, "pydimension")):
    sys.path.insert(0, _da_root)
from pydimension.data_preprocessing import (
    DataPreprocessor,
    DataPreprocessingConfig)

import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["dP_L", "v", "mu", "rho", "d", "phi"]
VARIABLE_UNITS = ["Pa/m", "m/s", "Pa·s", "kg/m³", "m", "-"]

# Dimension matrix, rows = (Mass, Length, Time), cols = VARIABLE_NAMES.
#   dP_L  [Pa/m]    = kg·m⁻²·s⁻²              ( 1, -2, -2)
#   v     [m/s]                                ( 0,  1, -1)
#   mu    [Pa·s]    = kg·m⁻¹·s⁻¹              ( 1, -1, -1)
#   rho   [kg/m³]                              ( 1, -3,  0)
#   d     [m]                                  ( 0,  1,  0)
#   phi   [-]       (dimensionless)            ( 0,  0,  0)
DIMENSION_MATRIX = np.array([
    # dP_L  v   mu  rho  d   phi
    [   1,  0,  1,   1,  0,  0],   # Mass
    [  -2,  1, -1,  -3,  1,  0],   # Length
    [  -2, -1, -1,   0,  0,  0],   # Time
], dtype=float)
DIMENSION_NAMES = ["Mass", "Length", "Time"]

# Reference exponent vectors over the 6 variables (dP_L, v, mu, rho, d, phi).
#   f      = dP_L^1 · v^-2 · rho^-1 · d^1
KNOWN_F_EXPONENTS  = np.array([ 1.0, -2.0,  0.0, -1.0,  1.0, 0.0])
#   Re_p   = v · rho · d · mu^-1
KNOWN_RE_EXPONENTS = np.array([ 0.0,  1.0, -1.0,  1.0,  1.0, 0.0])


# ──────────────────────────────────────────────────────────────────────────────
# Dimensional analysis — Stage-0 reduction
# ──────────────────────────────────────────────────────────────────────────────

# Pi-basis discovery is always done via
# pydimension.data_preprocessing.DataPreprocessor (see
# run_repo_dimensional_analysis).  No inline fallback path exists.


def format_pi_expression(basis_col: np.ndarray, names) -> str:
    parts = []
    for name, exp in zip(names, basis_col):
        if abs(exp) < 1e-10:
            continue
        if abs(exp - 1.0) < 1e-10:
            parts.append(f"{name}")
        elif abs(exp + 1.0) < 1e-10:
            parts.append(f"{name}^-1")
        elif abs(exp - round(exp)) < 1e-10:
            parts.append(f"{name}^{int(round(exp))}")
        else:
            parts.append(f"{name}^({exp:+.3f})")
    return " · ".join(parts) if parts else "1"


# ──────────────────────────────────────────────────────────────────────────────
# Repository pipeline: drive pydimension.data_preprocessing.DataPreprocessor
# ──────────────────────────────────────────────────────────────────────────────

def _write_dimension_matrix_csv(out_path: str, variable_names, dim_matrix: np.ndarray) -> str:
    """Emit a Dimension/Variable CSV that DataPreprocessor.load_dimension_matrix understands.

    ``dim_matrix`` is (n_dims, n_vars) with rows ordered (Mass, Length, Time).
    The repo's loader expects a column named ``Dimension`` plus one column per
    variable.  Passing this CSV bypasses the unit-string parser entirely.
    """
    import csv as _csv
    dim_names = ["Mass", "Length", "Time"][: dim_matrix.shape[0]]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["Dimension"] + list(variable_names))
        for i, dn in enumerate(dim_names):
            w.writerow([dn] + [int(dim_matrix[i, j]) for j in range(len(variable_names))])
    return out_path


def run_repo_dimensional_analysis(csv_path: str, input_vars, output_var: str,
                                  dim_matrix: np.ndarray, output_dir: str) -> dict:
    """Drive the repo's ``DataPreprocessor.process_with_dimensional_analysis``.

    We pass an explicit dimension-matrix CSV (built from the hand-checked
    integer matrix) so the result comes from pydimension's null-space +
    SymPy primitive-integer reduction, not from this script's inline copy.
    Returns a dict with the basis vectors, dimensionless expressions, and
    the ``afterDA`` dataframe of Pi groups.
    """
    os.makedirs(output_dir, exist_ok=True)
    dim_csv = os.path.join(output_dir, "dimension_matrix.csv")
    _write_dimension_matrix_csv(dim_csv, input_vars, dim_matrix)

    cfg = DataPreprocessingConfig(
        input_file=str(csv_path),
        input_variables=list(input_vars),
        output_variables=[output_var],
        dimension_matrix_file=dim_csv,
        normalize=True,
        normalize_basis=False,    # keep primitive integer basis vectors
        output_dir=output_dir)
    pre = DataPreprocessor(cfg)
    pre.process_with_dimensional_analysis(verbose=True)
    try:
        pre.save_dimensional_analysis_results()
    except Exception as e:
        print(f"  ⚠️ Could not save DataPreprocessor outputs: {e}")
    return {
        "preprocessor": pre,
        "basis_vectors": np.asarray(pre.basis_vectors, dtype=float),
        "expressions":   list(pre.dimensionless_expressions),
        "afterDA":       pre.afterDA_data,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(args):
    """Load LBM porous-media dataset.

    Expected columns (subset): dP_L, v, mu, rho, d, phi, f, Re_p, f_ergun,
    converged, stalled.
    """
    import csv
    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print(f"Loading LBM porous media data from {data_path}...")
    rows = []
    with open(data_path) as f:
        for r in csv.DictReader(f):
            if (r.get('converged', 'False') or 'False').strip().lower() != 'true':
                continue
            if (r.get('stalled', 'False') or 'False').strip().lower() == 'true':
                continue
            try:
                row = {
                    k: float(r[k])
                    for k in ['dP_L', 'v', 'mu', 'rho', 'd', 'phi',
                              'f', 'Re_p', 'f_ergun']
                }
                vals = list(row.values())
                if not all(np.isfinite(vals)):
                    continue
                if row['f'] <= 0 or row['v'] <= 0 or row['Re_p'] <= 0:
                    continue
                rows.append(row)
            except (ValueError, KeyError):
                continue

    if not rows:
        print(f"ERROR: No valid rows found in {data_path}")
        sys.exit(1)

    X = np.array([[r[k] for k in VARIABLE_NAMES] for r in rows])
    y = np.array([r['f'] for r in rows])
    Re_p = np.array([r['Re_p'] for r in rows])
    f_ergun = np.array([r['f_ergun'] for r in rows])

    print(f"  Loaded {X.shape[0]} converged LBM runs")
    print(f"  Variables: {VARIABLE_NAMES}")
    for i, name in enumerate(VARIABLE_NAMES):
        col = X[:, i]
        print(f"    {name:6s} range: [{col.min():.4g}, {col.max():.4g}]"
              f"  ({len(np.unique(np.round(col, 6)))} unique)")
    print(f"  Re_p range: [{Re_p.min():.3e}, {Re_p.max():.3e}]")
    print(f"  f range:    [{y.min():.3e}, {y.max():.3e}]")
    print(f"  f / f_ergun: mean = {(y / f_ergun).mean():.3f},  "
          f"std = {(y / f_ergun).std():.3f}")
    print()
    return X, y, Re_p, f_ergun


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Re_p, f_ergun, args):
    results = {"X_raw": X, "y_raw": y, "Re_p": Re_p, "f_ergun": f_ergun}

    # ───────── Step 0: Dimensional Analysis ─────────
    print("=" * 60)
    print("Step 0: Dimensional analysis (Buckingham-Pi reduction)")
    print("=" * 60)
    print(f"  Dimension matrix shape: {DIMENSION_MATRIX.shape}  "
          f"(rows = {DIMENSION_NAMES}, cols = {VARIABLE_NAMES})")
    rank = int(np.linalg.matrix_rank(DIMENSION_MATRIX))
    print(f"  Rank: {rank}   Expected Pi groups: "
          f"{DIMENSION_MATRIX.shape[1] - rank}")

    # Always use the repo's DataPreprocessor pipeline.
    repo_out_dir = os.path.join(args.output_dir, "_da_repo")
    print(f"  Using pydimension.data_preprocessing.DataPreprocessor "
          f"(output → {repo_out_dir})")
    repo_res = run_repo_dimensional_analysis(
        csv_path=args.data,
        input_vars=VARIABLE_NAMES,
        output_var="f",
        dim_matrix=DIMENSION_MATRIX,
        output_dir=repo_out_dir)
    pi_basis = repo_res["basis_vectors"]
    pi_expressions = repo_res["expressions"]
    results["da_repo"] = repo_res

    print(f"  Basis vectors shape: {pi_basis.shape}")
    for i in range(pi_basis.shape[1]):
        expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
        if pi_expressions and i < len(pi_expressions):
            print(f"    Pi{i+1} = {expr}    (repo: {pi_expressions[i]})")
        else:
            print(f"    Pi{i+1} = {expr}")

    # Project known exponents onto null space
    for label, ref in [("f (friction factor)", KNOWN_F_EXPONENTS),
                       ("Re_p (Reynolds)",     KNOWN_RE_EXPONENTS)]:
        coords, *_ = np.linalg.lstsq(pi_basis, ref, rcond=None)
        recon = pi_basis @ coords
        ref_n = ref / (np.linalg.norm(ref) + 1e-12)
        cos = float(np.dot(recon, ref_n) /
                    (np.linalg.norm(recon) + 1e-12))
        print(f"  Known {label} exponents in Pi span: cos = {cos:+.4f}  "
              f"(±1 means it lies in the discovered Pi space)")

    results["pi_basis"] = pi_basis

    # Pi features for Step 2: use externally-computed Re_p and phi.
    # (We EXCLUDE the f Pi group from inputs because it IS the target.)
    phi = X[:, 5]
    log10_Re = np.log10(np.maximum(Re_p, 1e-30))
    pi_features = np.column_stack([log10_Re, phi])
    fmin = pi_features.min(axis=0)
    fmax = pi_features.max(axis=0)
    rng = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    pi_features_norm = (pi_features - fmin) / rng

    results["pi_features"] = pi_features_norm
    results["pi_feature_names"] = ["log10(Re_p)", "phi"]
    print(f"  Pi features for Step 2 encoder: "
          f"{results['pi_feature_names']}")
    print(f"  pi_features_norm shape: {pi_features_norm.shape}  "
          f"range: [{pi_features_norm.min():.3f}, "
          f"{pi_features_norm.max():.3f}]")

    # Step 3 input: the Pi VALUES themselves, geometric-mean-centred per
    # column (a purely multiplicative rescaling, no min-max).  The scaling
    # encoder's internal log then sees centred log-Pi coordinates, and the
    # generators live in dimensionless (Re_p, phi) space.
    log10_pi_vals = np.column_stack([log10_Re,
                                     np.log10(np.maximum(phi, 1e-30))])
    log10_pi_vals = log10_pi_vals - log10_pi_vals.mean(axis=0, keepdims=True)
    pi_centred = 10.0 ** log10_pi_vals
    results["pi_centred"] = pi_centred
    results["feature_names_step3"] = ["Re_p", "phi"]
    print(f"  Centred Pi values for Step 3: shape {pi_centred.shape}  "
          f"range: [{pi_centred.min():.3g}, {pi_centred.max():.3g}]")
    print()

    # ───────── Step 1: Normalisation ─────────
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()

    # Log-transform y because f spans ~5 orders of magnitude.
    y_log = np.log10(np.maximum(y, 1e-30))

    # Step 3's three encoders apply X, X², and log(|X|.clamp(min=0.1)).  The
    # clamp threshold is calibrated for raw-physical X with multiplicative
    # spread (values O(1) ± some orders of magnitude).  If we min-max raw X
    # into [0, 1] first, many values land below 0.1 and get pinned by the
    # clamp -- the scaling encoder then loses its multiplicative signal and
    # rotational/translational win purely on input-preservation grounds,
    # regardless of the underlying physics.
    #
    # Fix: geometric-mean-centre each column (so per-column geometric mean
    # is exactly 1.0) and pass the result DIRECTLY -- no min-max.  Values
    # span roughly [0.05, 25] per column with the spread set by the raw
    # log-range, which is what the scaling encoder is calibrated for.
    if getattr(args, "log_normalize", True):
        log10_X = np.log10(np.maximum(X, 1e-30))
        col_std = log10_X.std(axis=0)
        active = col_std > 1e-8
        gmean_exp = np.zeros(log10_X.shape[1])
        gmean_exp[active] = log10_X[:, active].mean(axis=0)
        X_prescaled = 10 ** (log10_X - gmean_exp)
        # Min-max y only.
        ymin, ymax = y_log.min(), y_log.max()
        y_norm = (y_log - ymin) / max(ymax - ymin, 1e-12)
        X_norm_raw = X_prescaled
        norm_raw = {"X_normalized": X_norm_raw, "y_normalized": y_norm}
        print(f"  Log-prenormalisation ON  "
              f"(geometric-mean centring, NO min-max on X)")
        print(f"  X_prescaled range per col: "
              f"min={X_prescaled.min(axis=0)}, max={X_prescaled.max(axis=0)}")
    else:
        norm_raw = normalize_data(X, y_log, method="minmax")
        X_norm_raw = norm_raw["X_normalized"]
        y_norm = norm_raw["y_normalized"]
        print(f"  Log-prenormalisation OFF (plain min-max on raw X)")

    # Identify constant columns (their generators will be undetermined)
    raw_std = X.std(axis=0) / (np.abs(X.mean(axis=0)) + 1e-30)
    const_cols = [VARIABLE_NAMES[i] for i in range(len(VARIABLE_NAMES))
                  if raw_std[i] < 1e-6]
    if const_cols:
        print(f"  ⚠️  Constant/near-constant columns: {const_cols}")
        print(f"     Their symmetry exponents will be unconstrained "
              f"by the dataset.")

    print(f"  X_raw  range: [{X_norm_raw.min():.3f}, "
          f"{X_norm_raw.max():.3f}]  ({X_norm_raw.shape})")
    print(f"  y = log10(f) normalized to [0, 1]")
    print(f"  log10(f) range: [{y_log.min():.3f}, {y_log.max():.3f}]")
    print()

    # ───────── Step 2: Latent Dimension Discovery ─────────
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()

    # Feed pi_features directly to the encoder (raw_input=True skips the
    # [X, X², log|X|] augmentation, since these are already log Pi groups).
    enc_kwargs = {
        "encoder_hidden_dims": args.encoder_hidden,
        "raw_input": True,
    }
    print(f"  Step 2 encoder input: {pi_features_norm.shape[1]} Pi features "
          f"({results['pi_feature_names']})")
    print(f"  Multilayer encoder hidden dims: {args.encoder_hidden}")

    res_latent = discover_latent_dimension(
        pi_features_norm, y_norm,
        max_latent=min(3, pi_features_norm.shape[1]),
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts,
        seed=args.seed, **enc_kwargs)
    results["latent"] = res_latent
    n_latent = res_latent["optimal_n_latent"]
    print(f"\n  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        r2_tr = m.get("R2_train", float("nan"))
        print(f"    k={k}: R2_train={r2_tr:.4f}, "
              f"R2_test={m['R2']:.4f}, MSE={m['MSE']:.6f}")
    print()

    # ───────── Step 3: Symmetry Type Identification ─────────
    print("=" * 60)
    print("Step 3: Identifying symmetry type")
    print("=" * 60)
    sys.stdout.flush()
    names_step3 = results["feature_names_step3"]
    print(f"  Step 3 input: {pi_centred.shape[1]} geometric-mean-centred "
          f"Pi values ({names_step3})")
    print(f"  Generators therefore live in dimensionless (Re_p, phi) space.")

    res_sym = identify_symmetry(
        pi_centred, y_norm, n_latent=n_latent,
        decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs, n_restarts=args.n_restarts,
        seed=args.seed)
    results["symmetry"] = res_sym
    print(f"\n  Detected symmetry: {res_sym['symmetry_type']}")
    for stype, loss in sorted(res_sym["losses"].items(),
                              key=lambda kv: kv[1]):
        marker = " <--" if stype == res_sym["symmetry_type"] else ""
        print(f"    {stype:15s}: {loss:.6f}{marker}")
    sorted_losses = sorted(res_sym["losses"].values())
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        print(f"  Loss gap: {sorted_losses[1] / sorted_losses[0]:.1f}x")
    print()

    # ───────── Step 4: Generator Extraction ─────────
    print("=" * 60)
    print("Step 4: Extracting Lie-algebra generators")
    print("=" * 60)
    winner_type = res_sym["symmetry_type"]
    winner_encoder = res_sym["encoders"][winner_type]
    generators = extract_generators(winner_type, winner_encoder)
    results["generators"] = generators
    results["winner_type"] = winner_type
    results["winner_encoder"] = winner_encoder

    print(f"  Symmetry type: {winner_type}")
    print(f"  Generators: {len(generators)}")
    print()

    # Winning encoder weight vector(s), reported in Pi space.
    # Deep-Darcy reference: f ≈ 150·(1−φ)²/(Re_p·φ³), so in log-Pi space
    #   d(log f) = −1 · d(log Re_p) + s · d(log φ),  s = −3 − 2φ̄/(1−φ̄)
    # evaluated at the mean porosity φ̄ (local power-law slope of (1−φ)²/φ³).
    W = winner_encoder.weight_matrix
    phi_bar = float(phi.mean())
    darcy_ref = np.array([-1.0, -3.0 - 2.0 * phi_bar / (1.0 - phi_bar)])
    name_w = max(7, max(len(n) for n in names_step3))
    print("=" * 60)
    print("  Winning encoder weight vector(s)  [Pi space]")
    print("=" * 60)
    print(f"  Darcy reference direction (at phi_bar={phi_bar:.3f}): "
          f"{np.round(darcy_ref, 3)}")
    for i in range(W.shape[0]):
        row = W[i]
        denom = np.linalg.norm(row) + 1e-12
        row_n = row / denom
        print(f"  Row {i+1} ({winner_type}):")
        header = "    " + "  ".join(f"{n:>{name_w}s}" for n in names_step3)
        raw_s = "    " + "  ".join(f"{v:+{name_w}.4f}" for v in row)
        normed = "    " + "  ".join(f"{v:+{name_w}.4f}" for v in row_n)
        print(header)
        print(f"  raw :{raw_s}")
        print(f"  L2-n:{normed}")
        ref_n = darcy_ref / np.linalg.norm(darcy_ref)
        cos = float(np.dot(row_n, ref_n))
        print(f"  cos<row, Darcy [dlogf/dlogRe, dlogf/dlogphi]> = {cos:+.4f}")
    print()

    # ───────── Step 5: Physical Interpretation ─────────
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-Pi space along which")
        print(f"  the friction factor f is preserved: simultaneously rescale")
        print(f"  Re_p and phi along g and the flow stays invariant")
        print(f"  (Darcy regime preserved).\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(names_step3):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts) if parts else '(all components < 0.05)'}")
                _interpret_generator(g, names_step3)
                print()
    else:
        print(f"  Winner is '{winner_type}', not scaling. Generators below are "
              f"null-space directions of W in Pi space; they do NOT correspond")
        print(f"  to log-space rescalings (which is the physically meaningful")
        print(f"  invariance for the Darcy power-law). Reporting them for")
        print(f"  completeness only.\n")
        for i, g in enumerate(generators):
            g_arr = np.asarray(g)
            if g_arr.ndim == 1:
                parts = [f"{name}:{g_arr[j]:+.3f}"
                         for j, name in enumerate(names_step3)
                         if abs(g_arr[j]) > 0.05]
                print(f"  Generator {i+1}: [{', '.join(parts) if parts else '(all components < 0.05)'}]")
            else:
                print(f"  Generator {i+1}: shape={g_arr.shape}, not flattened")
    print()

    return results


def _interpret_generator(g, names):
    abs_g = np.abs(g)
    dominant = int(np.argmax(abs_g))
    name = names[dominant]
    coupled = [(names[j], g[j]) for j in range(len(g))
               if j != dominant and abs(g[j]) > 0.05]
    if coupled:
        direction = "increase" if g[dominant] > 0 else "decrease"
        compensations = []
        for cname, cval in coupled:
            cdirection = "increase" if cval > 0 else "decrease"
            compensations.append(f"{cdirection} {cname}")
        print(f"    Meaning: {direction} {name} while "
              f"{', '.join(compensations)}")
        print(f"             to keep friction factor f invariant")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_ergun_collapse(X, y, Re_p, output_dir):
    """Plot f·φ³/(1-φ) vs Re_p/(1-φ) — should collapse onto 150/x + 1.75."""
    os.makedirs(output_dir, exist_ok=True)
    phi = X[:, 5]

    x_mod = Re_p / np.maximum(1 - phi, 1e-30)
    y_mod = y * phi**3 / np.maximum(1 - phi, 1e-30)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Colour-code by porosity bin
    phi_rounded = np.round(phi, 3)
    uniq = sorted(set(phi_rounded))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(uniq)))
    for c, phi_val in zip(cmap, uniq):
        mask = np.isclose(phi_rounded, phi_val, atol=0.01)
        ax.loglog(x_mod[mask], y_mod[mask], 'o',
                  color=c, label=f"φ ≈ {phi_val:.3f} (n={mask.sum()})",
                  alpha=0.8, markersize=7, markeredgecolor='black',
                  markeredgewidth=0.4)

    # Textbook Ergun curve
    x_range = np.logspace(np.log10(x_mod.min()), np.log10(x_mod.max()), 200)
    y_ergun = 150 / x_range + 1.75
    ax.loglog(x_range, y_ergun, 'k--', lw=2.2,
              label="Textbook Ergun: 150/x + 1.75")

    ax.set_xlabel(r"$Re_p \,/\, (1-\phi)$")
    ax.set_ylabel(r"$f \cdot \phi^3 \,/\, (1-\phi)$")
    ax.set_title("Ergun Collapse — LBM points vs Textbook Curve", fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "lbm_ergun_collapse.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ergun collapse figure saved to {out_path}")


def plot_pi_candidates(X, y, results, output_dir):
    """Pi-basis heatmap + scatter of log10(f) vs each log10(Pi_k)."""
    os.makedirs(output_dir, exist_ok=True)
    pi_basis = results["pi_basis"]
    n_pi = pi_basis.shape[1]

    X_pos = np.maximum(X, 1e-30)
    log10_pi = np.log10(X_pos) @ pi_basis
    log10_y = np.log10(np.maximum(y, 1e-30))

    fig = plt.figure(figsize=(5 * (n_pi + 1), 5))
    gs = fig.add_gridspec(1, n_pi + 1, width_ratios=[1.3] + [1.0] * n_pi,
                          wspace=0.35)
    fig.suptitle("Porous Media LBM — Dimensional Analysis & Reduced Pi Candidates", fontweight="bold")

    # Pi-basis heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pi_basis.T, cmap="RdBu_r",
                   vmin=-np.max(np.abs(pi_basis)),
                   vmax=np.max(np.abs(pi_basis)), aspect="auto")
    ax.set_xticks(range(len(VARIABLE_NAMES)))
    ax.set_xticklabels(VARIABLE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(n_pi))
    ax.set_yticklabels([f"Pi{i+1}" for i in range(n_pi)])
    ax.set_title("Pi-basis exponents")
    for i in range(n_pi):
        for j in range(len(VARIABLE_NAMES)):
            v = pi_basis[j, i]
            if abs(v) > 1e-10:
                txt = (f"{v:+.0f}" if abs(v - round(v)) < 1e-9
                       else f"{v:+.2f}")
                ax.text(j, i, txt, ha="center", va="center",
                        color=("white"
                               if abs(v) > 0.6 * np.max(np.abs(pi_basis))
                               else "black"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="exponent")

    # Pi panels
    for i in range(n_pi):
        ax = fig.add_subplot(gs[0, i + 1])
        xk = log10_pi[:, i]
        ax.scatter(xk, log10_y, c="#4C72B0", s=22, alpha=0.7,
                   edgecolors="black", linewidths=0.3)
        expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
        ax.set_xlabel(f"log₁₀(Pi{i+1})\n{expr}")
        ax.set_ylabel("log₁₀(f)")
        ax.set_title(f"Pi{i+1}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "lbm_pi_candidates.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pi candidates figure saved to {out_path}")


def plot_results(X, y, results, output_dir):
    """Two-panel summary: symmetry-type bar chart + latent-dim R²."""
    os.makedirs(output_dir, exist_ok=True)
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Porous Media LBM — Symmetry Discovery", fontweight="bold")

    # Panel 1: symmetry type bar chart
    ax = axes[0]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452"
              for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylim(0, max(losses) * 1.5)
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Symmetry Type  (winner: {sym_res['symmetry_type']})")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom")
    sorted_losses = sorted(losses)
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        gap = sorted_losses[1] / sorted_losses[0]
        ax.text(0.97, 0.97, f"Loss gap: {gap:.1f}×",
                ha="right", va="top", transform=ax.transAxes, color="#333333")

    # Panel 2: latent-dim R² curve
    ax = axes[1]
    lat_res = results["latent"]
    ks = sorted(lat_res["metrics"].keys())
    r2_train = [lat_res["metrics"][k].get("R2_train", float("nan"))
                for k in ks]
    r2_test = [lat_res["metrics"][k]["R2"] for k in ks]
    ax.plot(ks, r2_train, "o--", color="#4C72B0", lw=1.8, ms=7,
            label="R² train")
    ax.plot(ks, r2_test, "s-", color="#DD8452", lw=2.2, ms=8,
            label="R² test")
    k_star = lat_res["optimal_n_latent"]
    ax.axvline(k_star, color="grey", ls=":", lw=1.5,
               label=f"k* = {k_star}")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("R²")
    ax.set_title("Latent Dimension Discovery")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "lbm_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Symmetry-discovery figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Discover symmetry in porous media LBM data")
    parser.add_argument("--data", default="dataset_lbm_porous.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir",
                        default="output_porous_media_lbm_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+",
                        default=[64, 32])
    parser.add_argument("--log-normalize", dest="log_normalize",
                        action="store_true", default=True,
                        help="Geometric-mean centre each column and skip the "
                             "min-max step on X for Step 3 (default ON). "
                             "Required for the scaling encoder to see "
                             "multiplicatively-meaningful coordinates rather "
                             "than min-max-clipped affine ones.")
    parser.add_argument("--no-log-normalize", dest="log_normalize",
                        action="store_false",
                        help="Plain min-max on raw X for Step 3. Handicaps "
                             "the scaling encoder via log-clamp truncation; "
                             "for ablation / comparison only.")
    args = parser.parse_args()

    X, y, Re_p, f_ergun = load_data(args)
    results = run_pipeline(X, y, Re_p, f_ergun, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_ergun_collapse(X, y, Re_p, args.output_dir)
    plot_pi_candidates(X, y, results, args.output_dir)
    plot_results(X, y, results, args.output_dir)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    print(f"  Symmetry: {sym_type}")
    print(f"  Generators: {len(results['generators'])}")
    if sym_type == "scaling":
        print(f"  These generators show how the dimensionless groups Re_p")
        print(f"  and phi can be simultaneously rescaled while preserving")
        print(f"  the friction factor f -- the porous-media analogue of")
        print(f"  the Ergun scaling invariance, in Pi space.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
