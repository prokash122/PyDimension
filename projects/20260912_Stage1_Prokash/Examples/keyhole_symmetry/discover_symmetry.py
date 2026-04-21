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
    python discover_symmetry.py --data dataset_keyhole.csv --no-pi-input
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

def compute_pi_basis(dim_matrix: np.ndarray) -> np.ndarray:
    """Null-space basis of the dimension matrix (the reduced Pi candidates).

    Mirrors ``pydimension.data_preprocessing.preprocessor.DataPreprocessor``:
    scipy gives a numerical null-space, SymPy (if available) is used to
    recover a primitive integer basis.  The returned matrix has shape
    (n_variables, n_pi_groups); column k gives the exponents of Pi_k.
    """
    from scipy.linalg import null_space
    null_sp = null_space(dim_matrix)
    if null_sp.shape[1] == 0:
        raise ValueError("Dimension matrix has trivial null space — nothing to reduce.")

    try:
        from sympy import Matrix, ilcm, igcd
    except ImportError:
        return null_sp

    M = Matrix(dim_matrix.astype(int).tolist())
    ns = M.nullspace()
    if not ns:
        return null_sp

    primitives = []
    for v in ns:
        denom = [x.as_numer_denom()[1] for x in v if x != 0]
        scale = denom[0] if denom else 1
        for d in denom[1:]:
            scale = ilcm(scale, d)
        w = v * scale
        elems = [abs(int(x)) for x in w if x != 0]
        g = elems[0] if elems else 1
        for e in elems[1:]:
            g = igcd(g, e)
        if g > 1:
            w = w // g
        for x in w:
            if x != 0:
                if x < 0:
                    w = -w
                break
        primitives.append(np.array([float(x) for x in w]))
    return np.column_stack(primitives)


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
    pi_basis = compute_pi_basis(DIMENSION_MATRIX)
    print(f"  Basis vectors shape: {pi_basis.shape}")
    for i in range(pi_basis.shape[1]):
        expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
        print(f"    Pi{i+1} = {expr}")
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
    print(f"  Reduced candidates (pi_features) shape: {pi_features.shape}  "
          f"range: [{pi_features.min():.3f}, {pi_features.max():.3f}]")
    print()

    # --- Normalize ---
    print("=" * 60)
    print("Step 1: Normalizing data")
    print("=" * 60)
    sys.stdout.flush()
    norm = normalize_data(X, y, method="minmax")
    X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
    results["normalization"] = norm
    print(f"  X range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    # Wire in the reduced candidates and a multilayer encoder by default.
    enc_kwargs = {"encoder_hidden_dims": args.encoder_hidden}
    if not args.no_pi_input:
        # Reduced candidates computed from physical (always-positive) X; the
        # pi_features path injects them directly, side-stepping the library's
        # log-of-normalised-X step which would see zeros after min-max scaling.
        enc_kwargs["pi_features"] = pi_features

    print(f"  Multilayer encoder hidden dims: {args.encoder_hidden}")
    print(f"  Reduced-candidate input: "
          f"{'ENABLED (' + str(pi_features.shape[1]) + ' Pi features)' if not args.no_pi_input else 'disabled'}")

    res_latent = discover_latent_dimension(
        X_norm, y_norm, max_latent=4,
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        **enc_kwargs,
    )
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
    res_sym = identify_symmetry(
        X_norm, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed,
    )
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

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-space along which Ke is preserved.")
        print(f"  Physically: simultaneous rescaling of variables that keeps the physics invariant.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(VARIABLE_NAMES):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts)}")
                # Physical meaning
                _interpret_generator(g, i + 1)
                print()
    elif winner_type == "rotational" and generators:
        for i, g in enumerate(generators):
            print(f"  Generator {i+1} (antisymmetric matrix):")
            print(f"    {np.round(g, 4)}")
    else:
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{name}:{g[j]:+.3f}" for j, name in enumerate(VARIABLE_NAMES) if abs(g[j]) > 0.05]
                print(f"  Generator {i+1}: [{', '.join(parts)}]")
    print()

    return results


def _interpret_generator(g, idx):
    """Give a physical interpretation of a scaling generator."""
    # Find the dominant variable
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    name = VARIABLE_NAMES[dominant]

    # Find coupled variables (others that must change to preserve Ke)
    coupled = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g))
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

    fig = plt.figure(figsize=(5 * (n_pi + 1), 5))
    gs  = fig.add_gridspec(1, n_pi + 1, width_ratios=[1.3] + [1.0] * n_pi,
                           wspace=0.35)
    fig.suptitle("Keyhole — Dimensional Analysis & Reduced Pi Candidates",
                 fontsize=14, fontweight="bold")

    # --- Panel A: Pi basis heatmap -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(pi_basis.T, cmap="RdBu_r",
                   vmin=-np.max(np.abs(pi_basis)), vmax=np.max(np.abs(pi_basis)),
                   aspect="auto")
    ax.set_xticks(range(len(VARIABLE_NAMES)))
    ax.set_xticklabels(VARIABLE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(n_pi))
    ax.set_yticklabels([f"Pi{i+1}" for i in range(n_pi)])
    ax.set_title("Pi-basis exponents", fontsize=11)
    # annotate cells
    for i in range(n_pi):
        for j in range(len(VARIABLE_NAMES)):
            v = pi_basis[j, i]
            if abs(v) > 1e-10:
                ax.text(j, i, f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.2f}",
                        ha="center", va="center",
                        color="white" if abs(v) > 0.6 * np.max(np.abs(pi_basis)) else "black",
                        fontsize=9)
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
        ax.set_xlabel(f"log₁₀(Pi{i+1})\n{expr}", fontsize=10)
        ax.set_ylabel("e*", fontsize=10)
        ax.set_title(f"Reduced candidate Pi{i+1}", fontsize=11)
        ax.legend(fontsize=9, loc="best")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "keyhole_pi_candidates.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pi candidates figure saved to {out_path}")


def plot_results(X, y, results, output_dir):
    """Create a focused 3-panel figure."""
    os.makedirs(output_dir, exist_ok=True)
    Ke = results["Ke"]
    generators = results["generators"]
    winner_type = results["winner_type"]
    norm = results["normalization"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("Keyhole — Symmetry Discovery", fontsize=15, fontweight="bold")

    # --- Panel 1: Known Ke vs e* ---
    ax = axes[0]
    ax.scatter(Ke, y, c="#4C72B0", s=20, alpha=0.6, edgecolors="none")
    coeffs = np.polyfit(Ke, y, 2)
    Ke_fit = np.linspace(Ke.min(), Ke.max(), 200)
    ax.plot(Ke_fit, np.polyval(coeffs, Ke_fit), "r-", lw=2, label="polynomial fit")
    ss_res = np.sum((y - np.polyval(coeffs, Ke))**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    ax.set_xlabel("Ke (known keyhole number)", fontsize=11)
    ax.set_ylabel("e*", fontsize=11)
    ax.set_title(f"Known Ke vs e*   (R² = {r2:.3f})", fontsize=12)
    ax.legend(fontsize=9)

    # --- Panel 2: Symmetry type identification ---
    ax = axes[1]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE", fontsize=11)
    ax.set_title(f"Symmetry Type (winner: {sym_res['symmetry_type']})", fontsize=12)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=9)

    # --- Panel 3: Generator orbits in log-space ---
    ax = axes[2]
    if generators and winner_type == "scaling":
        # Pick two most important variables from the first generator
        g = generators[0]
        importance = np.abs(g)
        top2 = np.argsort(importance)[-2:][::-1]
        d0, d1 = top2[0], top2[1]

        sc = ax.scatter(np.log10(X[:, d0] + 1e-12), np.log10(X[:, d1] + 1e-12),
                        c=y, cmap="plasma", s=15, alpha=0.5, edgecolors="none")
        fig.colorbar(sc, ax=ax, label="e*", fraction=0.046, pad=0.04)

        # Trace multiple orbits
        orbit_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        rng = np.random.default_rng(42)
        start_indices = rng.choice(len(X), min(4, len(X)), replace=False)

        for k, idx in enumerate(start_indices):
            x_start = norm["X_normalized"][idx]
            n_steps = 150
            eps = 0.02
            fwd = generator_orbit(x_start, g, n_steps, eps, winner_type)
            back = generator_orbit(x_start, g, n_steps, -eps, winner_type)
            orb = np.vstack([back[::-1], fwd[1:]])
            orb_orig = norm["scaler_X"].inverse_transform(orb)

            ax.plot(np.log10(np.abs(orb_orig[:, d0]) + 1e-12),
                    np.log10(np.abs(orb_orig[:, d1]) + 1e-12),
                    color=orbit_colors[k % len(orbit_colors)], lw=2, alpha=0.8,
                    label=f"orbit {k+1}" if k < 3 else None)

        ax.set_xlabel(f"log₁₀({VARIABLE_NAMES[d0]})", fontsize=11)
        ax.set_ylabel(f"log₁₀({VARIABLE_NAMES[d1]})", fontsize=11)
        ax.set_title("Generator Orbits (scaling directions)", fontsize=12)
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, f"No scaling orbits\n(detected: {winner_type})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Generator Orbits")

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
    parser.add_argument("--no-pi-input", action="store_true",
                        help="Disable the reduced (Pi) candidate features — "
                             "run the encoder on raw variables only.")
    args = parser.parse_args()

    X, y, Ke = load_data(args)
    results = run_pipeline(X, y, Ke, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
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
        print(f"  These generators show how physical variables can be")
        print(f"  simultaneously rescaled while preserving Ke and e*.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
