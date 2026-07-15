"""
Discover the symmetry structure of the known keyhole number using Stage1.

Physics
-------
The keyhole eccentricity e* in laser welding is governed by the known
dimensionless keyhole number (Eq. 12 in the paper):

    Ke = etaP / ((Tl-T0) * pi * rho * Cp * sqrt(alpha * Vs * r0^3))

This example runs the full PyDimension dimensionless-learning flow:

  0. Dimensional analysis: build the (M, L, T, K) dimension matrix of the
     seven physical inputs, compute the null-space basis, simplify it to
     a primitive integer basis (the "reduced candidates") via SymPy.
     With 7 variables and 4 fundamental dimensions, 7 - 4 = 3 independent
     Pi groups are produced — any valid Ke-like combination lies in their
     span.

  1. Feed the reduced candidates (log10 of each Pi group) into a
     multilayer encoder alongside the raw variables, and let Stage1
     discover the intrinsic latent dimension of e*.

  2. Identify the symmetry type of e* (translational / rotational /
     scaling) by competitive encoder training on raw X.

  3. Extract the Lie-algebra generators of the winning symmetry group
     and visualize which variable rescalings preserve the keyhole number.

Usage
-----
    python discover_symmetry.py --data dataset_keyhole.csv
    python discover_symmetry.py --data dataset_keyhole.csv --encoder-hidden 128 64
    python discover_symmetry.py --data dataset_keyhole.csv --no-pi-only
"""

import sys
import os
import argparse
import traceback
import multiprocessing

import numpy as np
import torch

# Add the Stage1 project to the path — try multiple locations
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
    "font.size":             19,
    "axes.titlesize":        26,
    "axes.labelsize":        24,
    "xtick.labelsize":       22,
    "ytick.labelsize":       22,
    "legend.fontsize":       18,
    "legend.title_fontsize": 20,
    "figure.titlesize":      30,
})

try:
    from preprocessing.normalize import normalize_data
    from intrinsic_coordinate.discovery import discover_latent_dimension
    from symmetry_discovery.identification import identify_symmetry
    from symmetry_discovery.generators import extract_generators, generator_orbit
except ImportError as e:
    print(f"ERROR: Could not import Stage1 modules: {e}")
    print(f"Copy preprocessing/, intrinsic_coordinate/, symmetry_discovery/ from")
    print(f"projects/20260912_Stage1_Prokash/ into the same directory as this script.")
    sys.exit(1)

# Required: the repository's dimensional-analysis pipeline.  Install
# pydimension (and its `seaborn` dependency) if the import below fails —
# this is the only Pi-discovery path the script supports.
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

# Prevent silent multiprocessing crashes on Windows
import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["etaP", "Vs", "r0", "alpha", "rho", "cp", "Tl-T0"]
VARIABLE_UNITS = ["W", "m/s", "m", "m²/s", "kg/m³", "J/(kg·K)", "K"]

# Dimension matrix, rows = (Mass, Length, Time, Temperature), cols = VARIABLE_NAMES.
#   etaP  [W]       = kg · m² · s⁻³        →  ( 1,  2, -3,  0)
#   Vs    [m/s]                             →  ( 0,  1, -1,  0)
#   r0    [m]                               →  ( 0,  1,  0,  0)
#   alpha [m²/s]                            →  ( 0,  2, -1,  0)
#   rho   [kg/m³]                           →  ( 1, -3,  0,  0)
#   cp    [J/(kg·K)]= m² · s⁻² · K⁻¹        →  ( 0,  2, -2, -1)
#   Tl-T0 [K]                               →  ( 0,  0,  0,  1)
DIMENSION_MATRIX = np.array([
    [1, 0, 0, 0,  1, 0, 0],   # Mass
    [2, 1, 1, 2, -3, 2, 0],   # Length
    [-3, -1, 0, -1, 0, -2, 0],  # Time
    [0, 0, 0, 0, 0, -1, 1],   # Temperature
], dtype=float)
DIMENSION_NAMES = ["Mass", "Length", "Time", "Temperature"]

# Known keyhole number exponents (Eq. 12):
#   Ke = etaP^1 * Vs^(-0.5) * r0^(-1.5) * alpha^(-0.5) * rho^(-1) * cp^(-1) * (Tl-T0)^(-1)
KNOWN_KE_EXPONENTS = np.array([1.0, -0.5, -1.5, -0.5, -1.0, -1.0, -1.0])


def compute_ke(X: np.ndarray) -> np.ndarray:
    """Compute the known keyhole number Ke from 7 physical variables."""
    etaP, Vs, r0, alpha, rho, cp, Tl_T0 = [X[:, i] for i in range(7)]
    return etaP / (Tl_T0 * np.pi * rho * cp * np.sqrt(alpha * Vs * r0**3))


# ──────────────────────────────────────────────────────────────────────────────
# Dimensional analysis — Stage-0 reduction to dimensionless candidates
# ──────────────────────────────────────────────────────────────────────────────

# Pi-basis discovery is always done via
# pydimension.data_preprocessing.DataPreprocessor (see
# run_repo_dimensional_analysis).  No inline fallback path exists.


def compute_pi_features(X_raw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Evaluate each Pi group on positive raw X, return log10 then min-max to [0, 1].

    The encoder expects augmented features on roughly the same magnitude as
    the other inputs, so we log-compress (the physical range spans many
    decades) and rescale to [0, 1] per column.
    """
    X_pos = np.maximum(X_raw, 1e-30)
    log_pi = np.log10(X_pos) @ basis                   # (n_samples, n_groups)
    log_pi = np.nan_to_num(log_pi, nan=0.0, posinf=0.0, neginf=0.0)
    mn = log_pi.min(axis=0, keepdims=True)
    mx = log_pi.max(axis=0, keepdims=True)
    rng = np.where(mx - mn > 1e-12, mx - mn, 1.0)
    return (log_pi - mn) / rng


def compute_pi_values_centred(X_raw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Per-sample Pi-group values, geometric-mean-centred per column.

    Unlike compute_pi_features (log10 + min-max, for the Step 2 MLP), this
    returns the Pi values themselves divided by their per-column geometric
    mean — a purely multiplicative rescaling. Step 3's competing encoders
    (X, X², log|X|) therefore act on genuinely multiplicative quantities,
    and the scaling encoder's internal log sees centred log-Pi coordinates.
    """
    X_pos = np.maximum(X_raw, 1e-30)
    log10_pi = np.log10(X_pos) @ basis
    log10_pi = log10_pi - log10_pi.mean(axis=0, keepdims=True)
    return 10.0 ** log10_pi


def format_pi_expression(basis_col: np.ndarray, names) -> str:
    """Human-readable product form of a single Pi group."""
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
    """Emit a Dimension/Variable CSV that DataPreprocessor.load_dimension_matrix understands."""
    import csv as _csv
    dim_names = ["Mass", "Length", "Time", "Temperature"][: dim_matrix.shape[0]]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["Dimension"] + list(variable_names))
        for i, dn in enumerate(dim_names):
            w.writerow([dn] + [int(dim_matrix[i, j]) for j in range(len(variable_names))])
    return out_path


def run_repo_dimensional_analysis(csv_path: str, input_vars, output_var: str,
                                  dim_matrix: np.ndarray, output_dir: str) -> dict:
    """Drive the repo's ``DataPreprocessor.process_with_dimensional_analysis``."""
    os.makedirs(output_dir, exist_ok=True)
    dim_csv = os.path.join(output_dir, "dimension_matrix.csv")
    _write_dimension_matrix_csv(dim_csv, input_vars, dim_matrix)
    cfg = DataPreprocessingConfig(
        input_file=str(csv_path),
        input_variables=list(input_vars),
        output_variables=[output_var],
        dimension_matrix_file=dim_csv,
        normalize=True,
        normalize_basis=False,
        output_dir=output_dir)
    pre = DataPreprocessor(cfg)
    pre.process_with_dimensional_analysis(verbose=True)
    return {
        "preprocessor": pre,
        "basis_vectors": np.asarray(pre.basis_vectors, dtype=float),
        "expressions":   list(pre.dimensionless_expressions),
        "afterDA":       pre.afterDA_data,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv_data(csv_path: str) -> dict:
    """Load keyhole data from CSV, skipping non-numeric columns."""
    import csv
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    input_cols = []
    for var in VARIABLE_NAMES:
        for i, h in enumerate(header):
            if h.strip() == var:
                input_cols.append(i)
                break

    output_col = None
    for target in ["e*", "Ke", "e"]:
        for i, h in enumerate(header):
            if h.strip() == target:
                output_col = i
                break
        if output_col is not None:
            break

    if len(input_cols) != 7:
        raise ValueError(f"Expected 7 input variables, found {len(input_cols)} in: {header}")
    if output_col is None:
        raise ValueError(f"Could not find output column (e*, Ke, or e) in: {header}")

    X = np.array([[float(rows[r][c]) for c in input_cols] for r in range(len(rows))])
    y = np.array([float(rows[r][output_col]) for r in range(len(rows))])
    print(f"  Loaded: {[header[i].strip() for i in input_cols]} -> {header[output_col].strip()}")
    return {"X": X, "y": y}


def load_data(args):
    """Load data from CSV, compute Ke, return (X, y, Ke)."""
    data_path = args.data
    if not os.path.exists(data_path):
        # Try relative to script directory
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Place your keyhole CSV (with columns {VARIABLE_NAMES} and e*/Ke)")
        print(f"in {_here}/ and run:")
        print(f"  python discover_symmetry.py --data <your_file.csv>")
        sys.exit(1)

    print(f"Loading keyhole data from {data_path}...")
    data = load_csv_data(data_path)

    X, y = data["X"], data["y"]
    Ke = compute_ke(X)

    print(f"  Samples: {X.shape[0]}")
    print(f"  Ke range: [{Ke.min():.4g}, {Ke.max():.4g}]")
    print(f"  e* range: [{y.min():.4f}, {y.max():.4f}]")
    print()
    return X, y, Ke


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Ke, args):
    """Run Stage1 symmetry discovery on the physical variables."""
    results = {"Ke": Ke}

    # --- Stage 0: Dimensional analysis → reduced Pi candidates ---
    print("=" * 60)
    print("Step 0: Dimensional analysis (Buckingham-Pi reduction)")
    print("=" * 60)
    print(f"  Dimension matrix shape: {DIMENSION_MATRIX.shape}  "
          f"(rows = {DIMENSION_NAMES}, cols = {VARIABLE_NAMES})")
    rank = int(np.linalg.matrix_rank(DIMENSION_MATRIX))
    print(f"  Rank: {rank}   Expected Pi groups: {DIMENSION_MATRIX.shape[1] - rank}")

    # Always use the repo's DataPreprocessor pipeline.
    repo_out_dir = os.path.join(args.output_dir, "_da_repo")
    os.makedirs(repo_out_dir, exist_ok=True)
    print(f"  Using pydimension.data_preprocessing.DataPreprocessor "
          f"(CSV: {args.data})")
    repo_res = run_repo_dimensional_analysis(
        csv_path=args.data,
        input_vars=VARIABLE_NAMES,
        output_var="e*",
        dim_matrix=DIMENSION_MATRIX,
        output_dir=repo_out_dir)
    pi_basis = repo_res["basis_vectors"]
    results["repo_da"] = repo_res
    print(f"  Basis vectors shape (repo): {pi_basis.shape}")
    for line in repo_res["expressions"]:
        print(f"    {line}")
    # Cosine similarity of each candidate (and of their combinations) to Ke,
    # as a sanity check that the known Ke lies in the null-space span.
    Ke_ref = KNOWN_KE_EXPONENTS / np.linalg.norm(KNOWN_KE_EXPONENTS)
    coords, *_ = np.linalg.lstsq(pi_basis, KNOWN_KE_EXPONENTS, rcond=None)
    recon = pi_basis @ coords
    recon_cos = float(np.dot(recon, Ke_ref) / (np.linalg.norm(recon) + 1e-12))
    print(f"  Known Ke exponents projected onto null-space basis: cos = {recon_cos:+.4f}  "
          f"(±1 means Ke lies in the Pi-group span)")
    pi_features = compute_pi_features(X, pi_basis)
    results["pi_basis"] = pi_basis
    results["pi_features"] = pi_features
    pi_feature_names = [f"Pi{i+1} (DA)" for i in range(pi_basis.shape[1])]
    results["pi_feature_names"] = pi_feature_names
    print(f"  Reduced candidates (pi_features) shape: {pi_features.shape}  "
          f"range: [{pi_features.min():.3f}, {pi_features.max():.3f}]")
    # Known Ke expressed in the discovered Pi basis (Ke = Π_k Pi_k^{c_k}):
    # `coords` from the lstsq above are exactly those exponents c_k.
    ke_pi_coords = coords
    results["ke_pi_coords"] = ke_pi_coords
    print(f"  Known Ke in Pi coordinates: Ke = "
          + " · ".join(f"Pi{i+1}^{c:.3g}" for i, c in enumerate(ke_pi_coords)))
    # Geometric-mean-centred Pi values: Step 3 input (multiplicative, no min-max).
    pi_centred = compute_pi_values_centred(X, pi_basis)
    results["pi_centred"] = pi_centred
    print(f"  Centred Pi values for Step 3: shape {pi_centred.shape}  "
          f"range: [{pi_centred.min():.3g}, {pi_centred.max():.3g}]")
    print()

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()
    pi_only = getattr(args, "pi_only", False)

    # Always normalize raw physical X — Step 3 (symmetry-type detection) runs
    # on physical variables so its translational/rotational/scaling encoders
    # (X, X², log|X|) act on multiplicatively-meaningful quantities.
    norm_raw = normalize_data(X, y, method="minmax")
    norm_raw["pi_only"] = pi_only
    X_norm_raw = norm_raw["X_normalized"]
    y_norm     = norm_raw["y_normalized"]

    # In pi-only mode, also normalize pi_features for Step 2.
    if pi_only:
        norm_pi = normalize_data(pi_features, y, method="minmax")
        X_norm_step2 = norm_pi["X_normalized"]
        print(f"  --pi-only: Step 2 input = {pi_features.shape[1]} dimensionless features; "
              f"Step 3 input = {pi_centred.shape[1]} geometric-mean-centred Pi values")
    else:
        X_norm_step2 = X_norm_raw

    n_pi = pi_centred.shape[1]
    pi_names_step3 = [f"Pi{i+1}" for i in range(n_pi)]
    results["normalization"] = norm_raw
    results["feature_names"] = pi_names_step3 if pi_only else VARIABLE_NAMES
    results["feature_names_step3"] = results["feature_names"]
    print(f"  X_raw range: [{X_norm_raw.min():.3f}, {X_norm_raw.max():.3f}]")
    if pi_only:
        print(f"  X_pi  range: [{X_norm_step2.min():.3f}, {X_norm_step2.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    # Wire in the reduced candidates and a multilayer encoder by default.
    enc_kwargs = {"encoder_hidden_dims": args.encoder_hidden}
    if pi_only:
        # Step 2 X is the Pi features; skip [X, X², log|X|] augmentation so
        # the encoder consumes the dimensionless groups directly.
        enc_kwargs["raw_input"] = True
        print(f"  --pi-only: Step 2 encoder input = {X_norm_step2.shape[1]} "
              f"dimensionless features (no [X, X², log|X|] augmentation)")
    else:
        # --no-pi-only: inject Pi features alongside [X, X², log|X|] augmentation.
        enc_kwargs["pi_features"] = pi_features

    print(f"  Multilayer encoder hidden dims: {args.encoder_hidden}")
    if not pi_only:
        print(f"  Reduced-candidate input: ENABLED ({pi_features.shape[1]} Pi features)")

    res_latent = discover_latent_dimension(
        X_norm_step2, y_norm, max_latent=4,
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        **enc_kwargs)
    results["latent"] = res_latent
    n_latent = res_latent["optimal_n_latent"]
    print(f"\n  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        r2_tr = m.get("R2_train", float("nan"))
        print(f"    k={k}: R2_train={r2_tr:.4f}, R2_test={m['R2']:.4f}, MSE={m['MSE']:.6f}")
    print()

    # --- Identify symmetry type ---
    print("=" * 60)
    print("Step 3: Identifying symmetry type")
    print("=" * 60)
    sys.stdout.flush()
    if pi_only:
        print(f"  Running Step 3 on the {n_pi} geometric-mean-centred Pi VALUES "
              f"(not the log-min-max Step 2 features).")
        print(f"  Centring is purely multiplicative, so the scaling encoder's "
              f"internal log sees centred log-Pi coordinates —")
        print(f"  no log-of-log degeneracy, and generators live in "
              f"dimensionless Pi space.")
        X_step3 = pi_centred
    else:
        X_step3 = X_norm_raw
    results["X_step3"] = X_step3
    res_sym = identify_symmetry(
        X_step3, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed)
    results["symmetry"] = res_sym
    print(f"\n  Detected symmetry: {res_sym['symmetry_type']}")
    for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
        marker = " <--" if stype == res_sym["symmetry_type"] else ""
        print(f"    {stype:15s}: {loss:.6f}{marker}")
    sorted_losses = sorted(res_sym["losses"].values())
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        print(f"  Loss gap: {sorted_losses[1] / sorted_losses[0]:.1f}x")
    print()

    # --- Extract generators ---
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

    # --- Report the winning encoder direction vs the known Ke Pi-exponents ---
    names_step3 = results["feature_names_step3"]
    W = winner_encoder.weight_matrix
    if pi_only and winner_type == "scaling":
        for i in range(W.shape[0]):
            row_n = W[i] / (np.linalg.norm(W[i]) + 1e-12)
            ref_n = ke_pi_coords / (np.linalg.norm(ke_pi_coords) + 1e-12)
            cos = float(np.dot(row_n, ref_n))
            print(f"  Encoder row {i+1} (L2-n): "
                  + ", ".join(f"{n}:{v:+.3f}" for n, v in zip(names_step3, row_n)))
            print(f"  cos<row, Ke Pi-exponents {np.round(ke_pi_coords, 3)}> = {cos:+.4f}  "
                  f"(±1 = the encoder rediscovered Ke)")
        print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-Pi space along which e* is preserved."
              if pi_only else
              f"  Each generator is a direction in log-space along which Ke is preserved.")
        print(f"  Simultaneously rescaling the Pi groups along g leaves the keyhole physics invariant.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(names_step3):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts)}")
                # Physical meaning
                _interpret_generator(g, i + 1, names_step3)
                print()
    elif winner_type == "rotational" and generators:
        for i, g in enumerate(generators):
            print(f"  Generator {i+1} (antisymmetric matrix):")
            print(f"    {np.round(g, 4)}")
    else:
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{name}:{g[j]:+.3f}" for j, name in enumerate(names_step3) if abs(g[j]) > 0.05]
                print(f"  Generator {i+1}: [{', '.join(parts)}]")
    print()

    return results


def _interpret_generator(g, idx, names):
    """Give a physical interpretation of a scaling generator."""
    # Find the dominant variable
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    name = names[dominant]

    # Find coupled variables (others that must change to preserve Ke)
    coupled = [(names[j], g[j]) for j in range(len(g))
               if j != dominant and abs(g[j]) > 0.05]

    if coupled:
        direction = "increase" if g[dominant] > 0 else "decrease"
        compensations = []
        for cname, cval in coupled:
            cdirection = "increase" if cval > 0 else "decrease"
            compensations.append(f"{cdirection} {cname}")
        print(f"    Meaning: {direction} {name} while {', '.join(compensations)}")
        print(f"             to keep Ke (and e*) unchanged")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels: Ke vs e*, symmetry losses, generator orbits)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pi_candidates(X, y, results, output_dir):
    """Plot the dimensional-analysis output: Pi-basis heatmap + y vs each Pi_k.

    This is the visual counterpart of the "reduced candidates" step: every Pi
    group discovered from the null-space of the dimension matrix gets its own
    scatter against the output, so the reader can see which ones collapse the
    data and which are under-determined.
    """
    os.makedirs(output_dir, exist_ok=True)
    pi_basis = results["pi_basis"]           # (n_vars, n_pi)
    n_pi = pi_basis.shape[1]

    # Raw (positive) Pi_k values from physical X — same as what gets log-scaled
    # and pushed into the encoder, but here we plot against y directly.
    X_pos = np.maximum(X, 1e-30)
    log10_pi = np.log10(X_pos) @ pi_basis     # (n_samples, n_pi)

    fig = plt.figure(figsize=(6 * (n_pi + 1), 6.5))
    gs  = fig.add_gridspec(1, n_pi + 1, width_ratios=[1.3] + [1.0] * n_pi,
                           wspace=0.35)
    fig.suptitle("Keyhole — Dimensional Analysis & Reduced Pi Candidates", fontweight="bold")

    # --- Panel A: Pi basis heatmap -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pi_basis.T, cmap="RdBu_r",
                   vmin=-np.max(np.abs(pi_basis)), vmax=np.max(np.abs(pi_basis)),
                   aspect="auto")
    ax.set_xticks(range(len(VARIABLE_NAMES)))
    ax.set_xticklabels(VARIABLE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(n_pi))
    ax.set_yticklabels([f"Pi{i+1}" for i in range(n_pi)])
    ax.set_title("Pi-basis exponents")
    # annotate cells
    for i in range(n_pi):
        for j in range(len(VARIABLE_NAMES)):
            v = pi_basis[j, i]
            if abs(v) > 1e-10:
                ax.text(j, i, f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.2f}",
                        ha="center", va="center",
                        color="white" if abs(v) > 0.6 * np.max(np.abs(pi_basis)) else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="exponent")

    # --- Panels B..: y vs log10(Pi_k) -----------------------------------------
    for i in range(n_pi):
        ax = fig.add_subplot(gs[0, i + 1])
        xk = log10_pi[:, i]
        # R² of a quadratic fit against the raw output
        try:
            coeffs = np.polyfit(xk, y, 2)
            yfit_on_data = np.polyval(coeffs, xk)
            ss_res = np.sum((y - yfit_on_data) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)
            xf = np.linspace(xk.min(), xk.max(), 200)
            ax.plot(xf, np.polyval(coeffs, xf), "r-", lw=1.8, alpha=0.9,
                    label=f"quad R²={r2:.2f}")
        except Exception:
            pass
        ax.scatter(xk, y, c="#4C72B0", s=18, alpha=0.7, edgecolors="none")
        expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
        ax.set_xlabel(f"log₁₀(Pi{i+1})\n{expr}")
        ax.set_ylabel("e*")
        ax.set_title(f"Reduced candidate Pi{i+1}")
        ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "keyhole_pi_candidates.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pi candidates figure saved to {out_path}")


def plot_discovered_law_and_generators(X, y, Ke, results, output_dir):
    """Visualize the discovered scaling law and generators in Pi space.

    Step 3 runs on the geometric-mean-centred Pi values, so everything here
    lives in dimensionless Pi coordinates:
      A. Discovered W (1 × n_pi) vs the known Ke Pi-exponents [0.5, 1, 1] —
         both L2-normalized. cos → ±1 means the encoder rediscovered Ke.
      B. Data collapse: e* vs the discovered latent z = W · log(Pi_centred).
      C. Heatmap of the null-space generators (rows) × Pi groups (cols).
      D. Orbit invariance: Pi(ε) = Pi0 · exp(ε·g). Because Ke = Π Pi_k^{c_k},
         Ke(ε)/Ke(0) = exp(ε · c·g) exactly; z is checked numerically with
         the encoder's log-clamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    winner_encoder = results["winner_encoder"]
    generators     = results["generators"]
    W              = winner_encoder.weight_matrix    # (k*, n_pi)
    pi_centred     = results["X_step3"]
    pi_names       = results["feature_names_step3"]
    ke_coords      = np.asarray(results["ke_pi_coords"], dtype=float)
    pi_basis       = results["pi_basis"]
    n_pi           = pi_centred.shape[1]
    n_gen          = len(generators)

    # ── Panel A: discovered direction vs known Ke Pi-exponents ─────────────
    W_dir = W[0] / (np.linalg.norm(W[0]) + 1e-12)
    Ke_dir = ke_coords / (np.linalg.norm(ke_coords) + 1e-12)
    if np.dot(W_dir, Ke_dir) < 0:
        W_dir = -W_dir
    cos_sim = float(np.dot(W_dir, Ke_dir))

    # ── Panel B: discovered latent (encoder-equivalent, incl. 0.1 clamp) ───
    log_pi = np.log(np.clip(np.abs(pi_centred), 0.1, None))
    z_disc = log_pi @ W[0]

    # ── Panel D: orbits in Pi space ─────────────────────────────────────────
    eps_grid = np.linspace(-0.5, 0.5, 41)
    orbit_Ke = np.zeros((n_gen, eps_grid.size))
    orbit_z  = np.zeros((n_gen, eps_grid.size))
    Pi_start = np.exp(np.log(pi_centred).mean(axis=0))     # geometric centre ≈ 1
    for k, g in enumerate(generators):
        Pi_orbit = Pi_start[None, :] * np.exp(np.outer(eps_grid, g))
        # Ke ratio is exact: log Ke moves by ε · (c · g)
        orbit_Ke[k] = np.exp(eps_grid * float(ke_coords @ g))
        orbit_z[k]  = np.log(np.clip(np.abs(Pi_orbit), 0.1, None)) @ W[0]
    z0 = float(np.log(np.clip(np.abs(Pi_start), 0.1, None)) @ W[0])

    # ── Build the figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(19, 14))
    gs  = fig.add_gridspec(2, 2, hspace=0.7, wspace=0.5)
    fig.suptitle("Keyhole — Discovered Law & Generators in Pi Space",
                 fontweight="bold")

    # Panel A
    ax = fig.add_subplot(gs[0, 0])
    x_pos = np.arange(n_pi)
    bar_w = 0.38
    ax.bar(x_pos - bar_w / 2, W_dir, bar_w,
           color="#4C72B0", edgecolor="black", label="Discovered W (encoder)")
    ax.bar(x_pos + bar_w / 2, Ke_dir, bar_w,
           color="#DD8452", edgecolor="black",
           label="Known Ke Pi-exponents")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pi_names)
    ax.set_ylabel("Exponent (L2-norm)")
    ax.set_title(f"Discovered W vs known Ke   (cos = {cos_sim:+.3f})")
    ax.text(0.5, -0.14,
            "Known: Ke = " + " · ".join(f"Pi{i+1}^{c:.2g}" for i, c in enumerate(ke_coords)),
            transform=ax.transAxes, ha="center", va="top", fontsize=17,
            color="#444444")
    ax.legend(loc="best", fontsize=15)

    # Panel B
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(z_disc, y, c="#4C72B0", s=22, alpha=0.75, edgecolors="none",
               label="e* vs discovered z")
    try:
        c = np.polyfit(z_disc, y, 2)
        y_hat = np.polyval(c, z_disc)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        zf = np.linspace(z_disc.min(), z_disc.max(), 200)
        ax.plot(zf, np.polyval(c, zf), "r-", lw=1.8, alpha=0.9,
                label=f"quad fit  R²={r2:.3f}")
    except Exception:
        pass
    ax.set_xlabel("Discovered latent  z = W · log(Pi_centred)")
    ax.set_ylabel("e*")
    ax.set_title("Data collapse onto the discovered law")
    ax.legend(loc="best", fontsize=15)

    # Panel C
    ax = fig.add_subplot(gs[1, 0])
    if n_gen > 0:
        G = np.stack([g if g.ndim == 1 else g.ravel() for g in generators], axis=0)
        vmax = float(np.max(np.abs(G))) or 1.0
        im = ax.imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_pi))
        ax.set_xticklabels(pi_names)
        ax.set_yticks(range(n_gen))
        ax.set_yticklabels([f"g{i+1}" for i in range(n_gen)])
        for i in range(n_gen):
            for j in range(n_pi):
                v = G[i, j]
                if abs(v) > 0.05:
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color="white" if abs(v) > 0.6 * vmax else "black",
                            fontsize=15)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{n_gen} generators of e*-invariance")
    else:
        ax.set_axis_off()
        ax.set_title("No generators")

    # Panel D
    ax = fig.add_subplot(gs[1, 1])
    cmap = plt.get_cmap("tab10")
    for k in range(n_gen):
        ax.plot(eps_grid, orbit_Ke[k], "-", color=cmap(k % 10), lw=1.8,
                label=f"Ke, g{k+1}")
    if n_gen > 0:
        # z0 sits at the geometric centre (≈ 0), so show 1 + Δz instead of a ratio
        ax.plot(eps_grid, 1.0 + (orbit_z[0] - z0), "k--", lw=2.2, alpha=0.9,
                label="1 + Δz (discovered)")
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_xlabel("Orbit parameter  ε   (Pi → Pi · exp(ε·g))")
    ax.set_ylabel("ratio to ε=0")
    ax.set_title("Invariance check along each orbit")
    ax.legend(loc="best", ncol=2, fontsize=14)
    y_dev = float(np.max(np.abs(orbit_Ke - 1.0))) if n_gen else 0.0
    ax.text(0.03, 0.03,
            f"max |ΔKe/Ke|={y_dev:.3f} at |ε|=0.5\n"
            "z stays flat (null(W) by construction)",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=14,
            color="#333333")

    # Footer: the actual Pi-group expressions behind the feature names.
    fig.text(0.05, 0.045,
             "    ".join(f"{n} = {format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)}"
                         for i, n in enumerate(pi_names)),
             ha="left", va="top", fontsize=15, color="#444444")

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_path = os.path.join(output_dir, "keyhole_discovered_law_generators.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Discovered-law & generators figure saved to {out_path}")


def plot_results(X, y, results, output_dir):
    """Two-panel summary: symmetry type bar chart + latent-dimension R² curve."""
    os.makedirs(output_dir, exist_ok=True)
    winner_type = results["winner_type"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Keyhole — Symmetry Discovery", fontweight="bold")

    # --- Panel 1: Symmetry type identification ---
    ax = axes[0]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylim(0, 0.05)
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

    # --- Panel 2: Latent dimension R² curve ---
    ax = axes[1]
    lat_res = results["latent"]
    ks = sorted(lat_res["metrics"].keys())
    r2_train = [lat_res["metrics"][k].get("R2_train", float("nan")) for k in ks]
    r2_test  = [lat_res["metrics"][k]["R2"] for k in ks]
    ax.plot(ks, r2_train, "o--", color="#4C72B0", lw=1.8, ms=7, label="R² train")
    ax.plot(ks, r2_test,  "s-",  color="#DD8452", lw=2.2, ms=8, label="R² test")
    k_star = lat_res["optimal_n_latent"]
    ax.axvline(k_star, color="grey", ls=":", lw=1.5, label=f"k* = {k_star}")
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("R²")
    ax.set_title("Latent Dimension Discovery")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "keyhole_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discover symmetry in keyhole welding data")
    parser.add_argument("--data", default="dataset_keyhole.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_keyhole_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[64, 32],
                        help="Hidden layer widths for the multilayer encoder "
                             "(default: 64 32)")
    parser.add_argument("--no-pi-only", action="store_true",
                        help="Disable the default pi-only mode: feed [X, X², log|X|, Pi] "
                             "to the Step 2 encoder instead of Pi groups alone.")
    args = parser.parse_args()
    args.pi_only = not args.no_pi_only

    X, y, Ke = load_data(args)
    results = run_pipeline(X, y, Ke, args)

    # Save artifacts for downstream generator plots (mirrors the concrete example).
    if args.pi_only and results["winner_type"] == "scaling":
        os.makedirs(args.output_dir, exist_ok=True)
        scaler_y = results["normalization"]["scaler_y"]
        np.savez(
            os.path.join(args.output_dir, "pipeline_artifacts.npz"),
            pi_centred=results["X_step3"],           # (n, n_pi) centred Pi values
            y=y,                                     # measured e*
            Ke=Ke,
            W=results["winner_encoder"].weight_matrix,   # scaling encoder (n_latent, n_pi)
            generators=np.array(results["generators"]),  # (n_pi - n_latent, n_pi), log-Pi
            ke_pi_coords=np.asarray(results["ke_pi_coords"], dtype=float),
            pi_names=np.array(results["feature_names_step3"]),
            # y minmax-normalization, to invert the decoder output back to e*
            y_min=np.asarray(getattr(scaler_y, "min_", 0.0), dtype=float),
            y_range=np.asarray(getattr(scaler_y, "range_", 1.0), dtype=float),
        )
        # Save the GENUINE trained model: winning scaling encoder + its jointly
        # trained decoder, so downstream plots run the real model end-to-end.
        winner = results["winner_type"]
        torch.save(
            {
                "encoder": results["symmetry"]["encoders"][winner],
                "decoder": results["symmetry"]["decoders"][winner],
                "symmetry_type": winner,
            },
            os.path.join(args.output_dir, "trained_model.pt"),
        )
        print(f"Artifacts saved to {args.output_dir}/pipeline_artifacts.npz")
        print(f"Trained model saved to {args.output_dir}/trained_model.pt")

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_pi_candidates(X, y, results, args.output_dir)
    plot_results(X, y, results, args.output_dir)
    if args.pi_only and results["winner_type"] == "scaling":
        plot_discovered_law_and_generators(X, y, Ke, results, args.output_dir)
    else:
        print("Skipping Pi-space law/generator figure "
              "(requires --pi-only and a scaling winner).")

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    print(f"  Symmetry: {sym_type}")
    print(f"  Generators: {len(results['generators'])}")
    if sym_type == "scaling":
        print(f"  These generators show how the dimensionless Pi groups can be")
        print(f"  simultaneously rescaled while preserving Ke and e*.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
