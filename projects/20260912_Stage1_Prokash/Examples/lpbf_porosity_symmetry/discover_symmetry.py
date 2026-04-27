"""
Discover the hidden scaling symmetry in Laser Powder Bed Fusion (LPBF)
porosity data using the PyDimension Stage1 pipeline.

Physics
-------
LPBF additive manufacturing defects — in particular the pore fraction `f`
left in the solidified track — are empirically controlled by a single
dimensionless "normalised enthalpy" group.  The same quantity was
extracted by hand in the accompanying notebook ``4_plot_3d-ZGAN(1).ipynb``
to collapse porosity curves from five different alloys (Al2024, Al6061,
Cu, SS304, Ti64) onto one master logistic curve:

    Pi = (Lv * rho * A * P * V) / (k^2 * (Tb - Tm)^2)

with seven physical inputs:

    | Variable                | Symbol | Units     | Dimensions        |
    |-------------------------|--------|-----------|-------------------|
    | Laser power             | P      | W         | kg·m²·s⁻³         |
    | Scan speed              | V      | m/s       | m·s⁻¹             |
    | Absorptivity            | A      | —         | dimensionless     |
    | Metal density           | rho    | kg/m³     | kg·m⁻³            |
    | Thermal conductivity    | k      | W/(m·K)   | kg·m·s⁻³·K⁻¹      |
    | Latent heat of vapour.  | Lv     | J/kg      | m²·s⁻²            |
    | Superheat (Tb-Tm)       | dT     | K         | K                 |

Four fundamental dimensions (M, L, T, K) in seven inputs give
``7 - 4 = 3`` independent dimensionless groups (Buckingham Pi).  ``Pi``
is one of them; for a single-output porosity problem one group suffices
to capture the leading-order collapse seen in the notebook.

The scaling symmetry is therefore: any continuous rescaling of
(P, V, A, rho, k, Lv, dT) that leaves ``Pi`` unchanged must also leave
the pore fraction unchanged.  The Stage1 pipeline recovers this
invariance directly from the experimental data without being told the
formula for ``Pi``.

This script runs the full PyDimension dimensionless-learning flow:

  0. Dimensional analysis: build the (M, L, T, K) dimension matrix of the
     seven LPBF inputs, compute the null-space basis, simplify to a
     primitive integer Pi basis via SymPy — these are the *reduced
     candidates* fed to the encoder.
  1. Feed the reduced candidates (log10 of each Pi group) into a
     multilayer encoder alongside the raw variables, and confirm pore
     fraction depends on a single latent (the normalised enthalpy Pi).
  2. Identify the symmetry type is **scaling** via competitive training.
  3. Extract the Lie-algebra generators of the scaling group — the
     simultaneous unit rescalings that preserve Pi and hence porosity.
  4. Produce a focussed 3-panel figure (Pi-collapse, symmetry-type bars,
     generator orbits).

Usage
-----
    python discover_symmetry.py --data dataset_lpbf.csv
    python discover_symmetry.py --data dataset_lpbf.csv --encoder-hidden 128 64
    python discover_symmetry.py --data dataset_lpbf.csv --no-pi-only
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

# Try to import the repository's DataPreprocessor for dimensional analysis.
# When available, this replaces the inline compute_pi_basis() so the example
# uses the same Buckingham-Pi pipeline as the rest of pydimension.
try:
    # The pydimension package lives at the repo root.  Walk up until we find it.
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
        DataPreprocessingConfig,
    )
    _REPO_DA_AVAILABLE = True
except ImportError as _da_err:
    print(f"  ⚠️ Could not import pydimension.data_preprocessing "
          f"({_da_err}); falling back to inline DA implementation.")
    _REPO_DA_AVAILABLE = False

# Prevent silent multiprocessing crashes on Windows
import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["P", "V", "A", "rho", "k", "Lv", "dT", "gamma", "Tb"]
VARIABLE_UNITS = ["W", "m/s", "-", "kg/m³", "W/(m·K)", "J/kg", "K", "N/m", "K"]

# Dimension matrix, rows = (Mass, Length, Time, Temperature), cols = VARIABLE_NAMES.
#   P      [W]       = kg · m² · s⁻³          →  ( 1,  2, -3,  0)
#   V      [m/s]                               →  ( 0,  1, -1,  0)
#   A      [-]       (dimensionless)           →  ( 0,  0,  0,  0)
#   rho    [kg/m³]                              →  ( 1, -3,  0,  0)
#   k      [W/(m·K)] = kg · m · s⁻³ · K⁻¹       →  ( 1,  1, -3, -1)
#   Lv     [J/kg]    = m² · s⁻²                 →  ( 0,  2, -2,  0)
#   dT     [K]                                  →  ( 0,  0,  0,  1)
#   gamma  [N/m]     = kg · s⁻²                 →  ( 1,  0, -2,  0)
#   Tb     [K]                                  →  ( 0,  0,  0,  1)
DIMENSION_MATRIX = np.array([
    # P  V  A  rho  k  Lv  dT  gamma  Tb
    [ 1, 0, 0,  1,  1,  0,  0,   1,   0],   # Mass
    [ 2, 1, 0, -3,  1,  2,  0,   0,   0],   # Length
    [-3,-1, 0,  0, -3, -2,  0,  -2,   0],   # Time
    [ 0, 0, 0,  0, -1,  0,  1,   0,   1],   # Temperature
], dtype=float)
DIMENSION_NAMES = ["Mass", "Length", "Time", "Temperature"]

# Pi (normalised enthalpy) known exponents, now over 9 variables
# (Lv·rho·A·P·V) / (k^2·dT^2) — γ and Tb have zero exponent here.
#                              P    V    A    rho    k    Lv   dT   γ    Tb
KNOWN_PI_EXPONENTS = np.array([1.0, 1.0, 1.0, 1.0, -2.0, 1.0, -2.0, 0.0, 0.0])

# ──────────────────────────────────────────────────────────────────────────────
# Per-material thermophysical properties used by the notebook's
# P_recoil / P_Laplace pressure ratio.  Copied verbatim from cell `db7b5b2e`
# of projects/20260912_Stage1_Prokash/Examples/4_plot_3d-ZGAN(1).ipynb.
# ──────────────────────────────────────────────────────────────────────────────
PRESSURE_PROPS = {
    # dHv [J/mol], T_boil [K], gamma [N/m]
    "Ti64":   dict(dHv=422000.0, Tb=3560.0, gamma=1.65),
    "SS304":  dict(dHv=341000.0, Tb=3090.0, gamma=1.80),
    "Al2024": dict(dHv=294000.0, Tb=2792.0, gamma=0.90),
    "Al6061": dict(dHv=294000.0, Tb=2792.0, gamma=0.90),
    "Cu":     dict(dHv=305000.0, Tb=2835.0, gamma=1.30),
}
# Physical constants used in the Clausius–Clapeyron form of P_recoil
P_ATM            = 101325.0      # Pa
R_GAS            = 8.314         # J/(mol·K)
R_KEYHOLE        = 35e-6         # m   (d_char/2 with d_char = 70 µm)
T_SURFACE_FACTOR = 1.05          # T_s = 1.05 · T_boil (notebook default)


def compute_pi(X: np.ndarray) -> np.ndarray:
    """Compute the notebook's normalised-enthalpy Pi from the 9-column X."""
    P, V, A, rho, k, Lv, dT = (X[:, i] for i in range(7))
    return (Lv * rho * A * P * V) / (k ** 2 * dT ** 2)


def compute_pressure_ratio(Tb: np.ndarray, gamma: np.ndarray, dHv: np.ndarray) -> np.ndarray:
    """P_recoil / P_Laplace from the notebook formula (non-power-law, per row).

    The exponential is the Clausius–Clapeyron expression used by the
    notebook's ``pressure_ratio()`` function in cell ``db7b5b2e`` of
    ``4_plot_3d-ZGAN(1).ipynb``.
    """
    T_s = T_SURFACE_FACTOR * Tb
    P_recoil  = 0.54 * P_ATM * np.exp((dHv / R_GAS) * (1.0 / Tb - 1.0 / T_s))
    P_laplace = 2.0 * gamma * np.cos(0.0) / R_KEYHOLE
    return P_recoil / P_laplace


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
    """Evaluate each Pi group on positive raw X, return log10 then min-max to [0, 1]."""
    X_pos = np.maximum(X_raw, 1e-30)
    log_pi = np.log10(X_pos) @ basis
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
# Repository pipeline: drive pydimension.data_preprocessing.DataPreprocessor
# ──────────────────────────────────────────────────────────────────────────────

def _enrich_lpbf_csv(src_path: str, dst_path: str) -> str:
    """Copy the dataset and append per-row gamma, Tb columns from PRESSURE_PROPS.

    The original ``dataset_lpbf.csv`` only has the seven physical inputs; the
    repo's ``DataPreprocessor`` reads variables straight from columns of the
    file, so for the 9-variable analysis we materialise a CSV that has the
    extra γ and Tb columns looked up per material.
    """
    import csv as _csv
    rows = []
    with open(src_path, "r") as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            mat = (r.get("source") or "").strip()
            if mat in PRESSURE_PROPS:
                r["gamma"] = PRESSURE_PROPS[mat]["gamma"]
                r["Tb"]    = PRESSURE_PROPS[mat]["Tb"]
                rows.append(r)
    if not rows:
        raise ValueError(f"No rows with a known material were found in {src_path}.")
    fieldnames = list(rows[0].keys())
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    with open(dst_path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return dst_path


def _write_dimension_matrix_csv(out_path: str, variable_names, dim_matrix: np.ndarray) -> str:
    """Emit a Dimension/Variable CSV that DataPreprocessor.load_dimension_matrix understands.

    ``dim_matrix`` is (n_dims, n_vars) with rows ordered
    (Mass, Length, Time, Temperature).  The repo's loader expects a column
    named ``Dimension`` plus one column per variable.
    """
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
    """Drive the repo's ``DataPreprocessor.process_with_dimensional_analysis``.

    We pass an explicit dimension-matrix CSV (built from our hand-checked
    integer matrix) so the result doesn't depend on the unit-string parser
    in ``DataPreprocessor`` — that parser mishandles e.g. ``W/(m·K)``.
    Returns a dict with the basis vectors, dimensionless expressions, and
    the ``afterDA`` dataframe of Pi groups.
    """
    if not _REPO_DA_AVAILABLE:
        raise RuntimeError("pydimension.data_preprocessing is not importable")
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
        output_dir=output_dir,
    )
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
    """Load LPBF porosity data from CSV.

    Expected columns: source, P, V, A, rho, k, Lv, dT, Pore.  The
    per-material γ (surface tension), Tb (boiling temperature), and
    dHv (molar heat of vaporisation) are looked up from
    :data:`PRESSURE_PROPS` using the ``source`` column, so X is returned
    with 9 columns matching :data:`VARIABLE_NAMES`.
    """
    import csv
    CSV_VARS = ["P", "V", "A", "rho", "k", "Lv", "dT"]  # columns in the file
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    def _col(name):
        for i, h in enumerate(header):
            if h.strip() == name:
                return i
        return None

    input_cols = [_col(v) for v in CSV_VARS]
    if any(c is None for c in input_cols):
        missing = [v for v, c in zip(CSV_VARS, input_cols) if c is None]
        raise ValueError(f"CSV is missing required columns: {missing}  (header={header})")
    source_col = _col("source")
    if source_col is None:
        raise ValueError(f"CSV must have a 'source' column to look up gamma / Tb: {header}")
    output_col = None
    for target in ["Pore", "pore", "pore_fraction", "f", "porosity"]:
        output_col = _col(target)
        if output_col is not None:
            break
    if output_col is None:
        raise ValueError(f"Could not find output column (Pore/porosity) in: {header}")

    X_list, y_list, mat_list, dHv_list = [], [], [], []
    for r in rows:
        try:
            mat = r[source_col].strip()
            if mat not in PRESSURE_PROPS:
                continue
            props = PRESSURE_PROPS[mat]
            row_vals = [float(r[c]) for c in input_cols]
            # Append gamma and Tb as the 8th and 9th variables
            row_vals.extend([props["gamma"], props["Tb"]])
            X_list.append(row_vals)
            y_list.append(float(r[output_col]))
            mat_list.append(mat)
            dHv_list.append(props["dHv"])
        except (ValueError, IndexError):
            continue
    X = np.array(X_list)
    y = np.array(y_list)
    dHv = np.array(dHv_list)
    materials = np.array(mat_list)
    print(f"  Loaded: {CSV_VARS + ['gamma (from source)', 'Tb (from source)']} -> "
          f"{header[output_col].strip()}")
    print(f"  Unique materials: {sorted(set(mat_list))}")
    return {"X": X, "y": y, "materials": materials, "dHv": dHv}


def load_data(args):
    """Load data from CSV, compute Pi and P_recoil/P_Laplace.

    Returns
    -------
    X          : (n, 9)      physical inputs (P, V, A, rho, k, Lv, dT, γ, Tb)
    y          : (n,)        pore fraction, clipped to [0, 1]
    Pi         : (n,)        notebook's normalised enthalpy
    PR         : (n,)        P_recoil / P_Laplace ratio (non-power-law feature)
    materials  : (n,)        material name per row, for colouring the 3D plot
    """
    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        print(f"Place your LPBF CSV (with columns source, P, V, A, rho, k, Lv, dT, Pore)")
        print(f"in {_here}/ and run:")
        print(f"  python discover_symmetry.py --data <your_file.csv>")
        sys.exit(1)

    print(f"Loading LPBF porosity data from {data_path}...")
    data = load_csv_data(data_path)

    X         = data["X"]          # (n, 9)
    y         = data["y"]
    materials = data["materials"]
    dHv       = data["dHv"]

    Pi = compute_pi(X)
    # Columns 7, 8 are gamma, Tb
    PR = compute_pressure_ratio(Tb=X[:, 8], gamma=X[:, 7], dHv=dHv)

    # Drop rows with non-positive / non-finite Pi or PR (log10 will be taken later)
    mask = np.isfinite(Pi) & (Pi > 0) & np.isfinite(PR) & (PR > 0)
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with non-positive Pi/PR")
        X, y, Pi, PR, materials = X[mask], y[mask], Pi[mask], PR[mask], materials[mask]

    y = np.clip(y, 0.0, 1.0)

    print(f"  Samples: {X.shape[0]}")
    print(f"  Pi range: [{Pi.min():.4g}, {Pi.max():.4g}]")
    print(f"  PR (P_recoil/P_Laplace) range: [{PR.min():.4g}, {PR.max():.4g}]")
    print(f"  Pore range: [{y.min():.4f}, {y.max():.4f}]")
    print()
    return X, y, Pi, PR, materials


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Pi, PR, materials, args):
    """Run Stage1 symmetry discovery on the LPBF physical variables."""
    results = {"Pi": Pi, "PR": PR, "X_raw": X, "materials": materials}

    # --- Stage 0: Dimensional analysis → reduced Pi candidates ---
    print("=" * 60)
    print("Step 0: Dimensional analysis (Buckingham-Pi reduction)")
    print("=" * 60)
    print(f"  Dimension matrix shape: {DIMENSION_MATRIX.shape}  "
          f"(rows = {DIMENSION_NAMES}, cols = {VARIABLE_NAMES})")
    rank = int(np.linalg.matrix_rank(DIMENSION_MATRIX))
    print(f"  Rank: {rank}   Expected Pi groups: {DIMENSION_MATRIX.shape[1] - rank}")

    # Prefer the repo's DataPreprocessor pipeline.  We hand it the dataset
    # CSV (enriched with γ, Tb columns) and an explicit dimension-matrix
    # CSV so the result comes from pydimension's null-space + SymPy
    # primitive-integer reduction, not from this script's inline copy.
    if _REPO_DA_AVAILABLE and not getattr(args, "no_repo_da", False):
        repo_out_dir = os.path.join(args.output_dir, "_da_repo")
        enriched_csv = os.path.join(repo_out_dir, "dataset_lpbf_enriched.csv")
        os.makedirs(repo_out_dir, exist_ok=True)
        _enrich_lpbf_csv(args.data, enriched_csv)
        print(f"  Using pydimension.data_preprocessing.DataPreprocessor "
              f"(enriched CSV: {enriched_csv})")
        repo_res = run_repo_dimensional_analysis(
            csv_path=enriched_csv,
            input_vars=VARIABLE_NAMES,
            output_var="Pore",
            dim_matrix=DIMENSION_MATRIX,
            output_dir=repo_out_dir,
        )
        pi_basis = repo_res["basis_vectors"]
        results["repo_da"] = repo_res
        print(f"  Basis vectors shape (repo): {pi_basis.shape}")
        for line in repo_res["expressions"]:
            print(f"    {line}")
    else:
        if not _REPO_DA_AVAILABLE:
            print(f"  Falling back to inline DA (pydimension not importable)")
        pi_basis = compute_pi_basis(DIMENSION_MATRIX)
        print(f"  Basis vectors shape: {pi_basis.shape}")
        for i in range(pi_basis.shape[1]):
            expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
            print(f"    Pi{i+1} = {expr}")
    # Verify the known normalised-enthalpy Pi lies in the null-space span.
    coords, *_ = np.linalg.lstsq(pi_basis, KNOWN_PI_EXPONENTS, rcond=None)
    recon = pi_basis @ coords
    ref_n = KNOWN_PI_EXPONENTS / (np.linalg.norm(KNOWN_PI_EXPONENTS) + 1e-12)
    recon_cos = float(np.dot(recon, ref_n) / (np.linalg.norm(recon) + 1e-12))
    print(f"  Known Pi exponents projected onto null-space basis: cos = {recon_cos:+.4f}  "
          f"(±1 means the notebook Pi lies in the Pi-group span)")
    pi_features = compute_pi_features(X, pi_basis)

    # --- Empirical extra feature: P_recoil / P_Laplace --------------------
    # The notebook visualises pore fraction over (log10(Pi), log10(PR)).
    # PR is a Clausius–Clapeyron expression — non-power-law — so it cannot
    # be produced by Buckingham-Pi null-space reduction alone.  We inject
    # it here as a precomputed feature: log10(PR) min-max scaled to [0, 1].
    logPR = np.log10(np.maximum(PR, 1e-30))
    logPR_scaled = (logPR - logPR.min()) / (logPR.max() - logPR.min() + 1e-12)
    pi_features = np.hstack([pi_features, logPR_scaled.reshape(-1, 1)])
    pi_feature_names = [f"Pi{i+1} (DA)" for i in range(pi_basis.shape[1])] + [
        "log10(P_recoil/P_Laplace)"
    ]
    results["pi_basis"] = pi_basis
    results["pi_features"] = pi_features
    results["pi_feature_names"] = pi_feature_names
    print(f"  Reduced candidates (pi_features) shape: {pi_features.shape}  "
          f"range: [{pi_features.min():.3f}, {pi_features.max():.3f}]")
    print(f"  Feature list: {pi_feature_names}")
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
    if getattr(args, "log_normalize", False):
        # Geometric-mean centring so the scaling encoder (which applies
        # log(X.clamp(0.1)) internally) sees centred log-physical coordinates
        # instead of min-max-clipped affine ones.
        log10_X = np.log10(np.maximum(X, 1e-30))
        gmean_exp = log10_X.mean(axis=0)
        X_prescaled = 10 ** (log10_X - gmean_exp)
        norm_raw = normalize_data(X_prescaled, y, method="minmax")
        norm_raw["log_prescaled"] = True
        norm_raw["gmean_exp"] = gmean_exp
        print(f"  Log-prenormalisation enabled (geometric-mean centring)")
        print(f"  X_prescaled range: [{X_prescaled.min():.3g}, {X_prescaled.max():.3g}]")
    else:
        norm_raw = normalize_data(X, y, method="minmax")
        norm_raw["log_prescaled"] = False
    norm_raw["pi_only"] = pi_only
    X_norm_raw = norm_raw["X_normalized"]
    y_norm     = norm_raw["y_normalized"]

    # In pi-only mode, also normalize pi_features for Step 2.
    if pi_only:
        norm_pi = normalize_data(pi_features, y, method="minmax")
        X_norm_step2 = norm_pi["X_normalized"]
        print(f"  --pi-only: Step 2 input = {pi_features.shape[1]} dimensionless features; "
              f"Step 3 input = raw physical X ({X_norm_raw.shape[1]} variables)")
    else:
        X_norm_step2 = X_norm_raw

    results["normalization"] = norm_raw
    results["feature_names"] = VARIABLE_NAMES   # Step 3 always on physical X
    print(f"  X_raw range: [{X_norm_raw.min():.3f}, {X_norm_raw.max():.3f}]")
    if pi_only:
        print(f"  X_pi  range: [{X_norm_step2.min():.3f}, {X_norm_step2.max():.3f}]")
    print()

    # --- Discover latent dimension ---
    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    # Default pipeline: multilayer encoder + reduced Pi candidates.
    # The pi_features path injects the precomputed log10(Pi_k) groups
    # (one column per Buckingham-Pi basis vector) directly, side-stepping
    # the library's log-of-normalised-X step.  That matters here because
    # LPBF material-property columns (A, rho, k, Lv, dT) only take 5
    # discrete values and entire material groups map to 0 after min-max
    # scaling — log(0) and negative exponents would blow up to NaN.
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
        print(f"  Injecting {pi_features.shape[1]} reduced Pi candidate(s) "
              f"(log10 + min-max to [0, 1])")
    print(f"  Multilayer encoder hidden dims: {args.encoder_hidden}")

    res_latent = discover_latent_dimension(
        X_norm_step2, y_norm, max_latent=4,
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
    if pi_only:
        print(f"  Running Step 3 on raw physical X ({X_norm_raw.shape[1]} variables) "
              f"so the translational/rotational/scaling encoders see")
        print(f"  multiplicatively-meaningful quantities (avoids the log-of-log "
              f"degeneracy of feeding pre-log-scaled Pi groups).")
    res_sym = identify_symmetry(
        X_norm_raw, y_norm, n_latent=n_latent, decoder=res_latent["best_decoder"],
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

    # --- Report the winning encoder's weight vector ---
    # Step 3 always runs on physical X, so the encoder's columns map 1:1 to
    # VARIABLE_NAMES regardless of --pi-only.  For scaling, z = W · log|X|.
    W = winner_encoder.weight_matrix  # (n_latent, n_inputs)
    names_for_W = VARIABLE_NAMES
    print("=" * 60)
    print("  Winning encoder weight vector(s)")
    print("=" * 60)
    name_w = max(7, max(len(n) for n in names_for_W))
    for i in range(W.shape[0]):
        row = W[i]
        denom = np.linalg.norm(row) + 1e-12
        row_n = row / denom
        print(f"  Row {i+1} ({winner_type}):")
        header = "    " + "  ".join(f"{n:>{name_w}s}" for n in names_for_W)
        raw    = "    " + "  ".join(f"{v:+{name_w}.4f}" for v in row)
        normed = "    " + "  ".join(f"{v:+{name_w}.4f}" for v in row_n)
        print(header)
        print(f"  raw :{raw}")
        print(f"  L2-n:{normed}")
        # Compare direction against known Pi exponents.
        ref = KNOWN_PI_EXPONENTS
        ref_n = ref / np.linalg.norm(ref)
        cos = float(np.dot(row_n, ref_n))
        print(f"  cos<row, known-Pi-exponents> = {cos:+.4f}  "
              f"(±1 means perfect alignment)")
    print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-space along which Pi is preserved.")
        print(f"  Physically: simultaneous rescaling of variables that keeps the")
        print(f"  normalised enthalpy Pi (and therefore the pore fraction) invariant.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(VARIABLE_NAMES):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts)}")
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
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    name = VARIABLE_NAMES[dominant]

    coupled = [(VARIABLE_NAMES[j], g[j]) for j in range(len(g))
               if j != dominant and abs(g[j]) > 0.05]

    if coupled:
        direction = "increase" if g[dominant] > 0 else "decrease"
        compensations = []
        for cname, cval in coupled:
            cdirection = "increase" if cval > 0 else "decrease"
            compensations.append(f"{cdirection} {cname}")
        print(f"    Meaning: {direction} {name} while {', '.join(compensations)}")
        print(f"             to keep Pi (and pore fraction) unchanged")


# ──────────────────────────────────────────────────────────────────────────────
# Visualization (3 panels: Pi vs Pore, symmetry losses, generator orbits)
# ──────────────────────────────────────────────────────────────────────────────

def plot_pi_candidates(X, y, results, output_dir):
    """Plot the dimensional-analysis output: Pi-basis heatmap + Pore vs each Pi_k.

    This is the visual counterpart of the "reduced candidates" step: every Pi
    group discovered from the null-space of the dimension matrix gets its own
    scatter against Pore, so the reader can see which ones collapse the data
    and which are under-determined (e.g. Pi = A is constant within a material
    so it will look like a vertical stripe pattern).
    """
    os.makedirs(output_dir, exist_ok=True)
    pi_basis = results["pi_basis"]
    n_pi = pi_basis.shape[1]

    X_pos = np.maximum(X, 1e-30)
    log10_pi = np.log10(X_pos) @ pi_basis

    fig = plt.figure(figsize=(5 * (n_pi + 1), 5))
    gs  = fig.add_gridspec(1, n_pi + 1, width_ratios=[1.3] + [1.0] * n_pi,
                           wspace=0.35)
    fig.suptitle("LPBF Porosity — Dimensional Analysis & Reduced Pi Candidates",
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
    for i in range(n_pi):
        for j in range(len(VARIABLE_NAMES)):
            v = pi_basis[j, i]
            if abs(v) > 1e-10:
                ax.text(j, i, f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.2f}",
                        ha="center", va="center",
                        color="white" if abs(v) > 0.6 * np.max(np.abs(pi_basis)) else "black",
                        fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="exponent")

    # --- Panels B..: Pore fraction vs log10(Pi_k) ----------------------------
    from scipy.optimize import curve_fit
    def _logistic(x, k, x0):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    for i in range(n_pi):
        ax = fig.add_subplot(gs[0, i + 1])
        xk = log10_pi[:, i]
        ax.scatter(xk, y, c="#4C72B0", s=18, alpha=0.7, edgecolors="none")
        # If Pi_k is effectively constant (e.g. Pi = A, 5 discrete values)
        # skip fitting — the scatter already tells the story.
        if xk.max() - xk.min() > 1e-6:
            try:
                order = np.argsort(xk)
                x0_init = xk[order][np.argmin(np.abs(y[order] - 0.5))]
                popt, _ = curve_fit(_logistic, xk, y,
                                    p0=[4.0, x0_init],
                                    bounds=([0.1, xk.min() - 2], [50.0, xk.max() + 5]),
                                    maxfev=5000)
                xf = np.linspace(xk.min(), xk.max(), 200)
                ax.plot(xf, _logistic(xf, *popt), "r-", lw=1.8, alpha=0.9,
                        label="logistic fit")
                yhat = _logistic(xk, *popt)
                ss_res = np.sum((y - yhat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-12)
                ax.text(0.03, 0.95, f"R² = {r2:.2f}", transform=ax.transAxes,
                        va="top", ha="left", fontsize=10,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            except Exception:
                pass
            ax.legend(fontsize=9, loc="lower right")
        expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
        ax.set_xlabel(f"log₁₀(Pi{i+1})\n{expr}", fontsize=10)
        ax.set_ylabel("Pore fraction", fontsize=10)
        ax.set_title(f"Reduced candidate Pi{i+1}", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "lpbf_pi_candidates.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pi candidates figure saved to {out_path}")


def plot_results(X, y, results, output_dir):
    """Two-panel summary: symmetry type bar chart + latent-dimension R² curve."""
    os.makedirs(output_dir, exist_ok=True)
    winner_type = results["winner_type"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("LPBF Porosity — Symmetry Discovery", fontsize=15, fontweight="bold")

    # --- Panel 1: Symmetry type identification ---
    ax = axes[0]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE", fontsize=12)
    ax.set_title(f"Symmetry Type  (winner: {sym_res['symmetry_type']})", fontsize=13)
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=10)
    sorted_losses = sorted(losses)
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        gap = sorted_losses[1] / sorted_losses[0]
        ax.text(0.97, 0.97, f"Loss gap: {gap:.1f}\u00d7",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=10, color="#333333")

    # --- Panel 2: Latent dimension R² curve ---
    ax = axes[1]
    lat_res = results["latent"]
    ks = sorted(lat_res["metrics"].keys())
    r2_train = [lat_res["metrics"][k].get("R2_train", float("nan")) for k in ks]
    r2_test  = [lat_res["metrics"][k]["R2"] for k in ks]
    ax.plot(ks, r2_train, "o--", color="#4C72B0", lw=1.8, ms=7, label="R\u00b2 train")
    ax.plot(ks, r2_test,  "s-",  color="#DD8452", lw=2.2, ms=8, label="R\u00b2 test")
    k_star = lat_res["optimal_n_latent"]
    ax.axvline(k_star, color="grey", ls=":", lw=1.5, label=f"k* = {k_star}")
    ax.set_xlabel("Latent dimension k", fontsize=12)
    ax.set_ylabel("R\u00b2", fontsize=12)
    ax.set_title("Latent Dimension Discovery", fontsize=13)
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "lpbf_porosity_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 3D surface plot (matches notebook 4_plot_3d-ZGAN.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_surface(Pi, PR, y, materials, output_dir):
    """Recreate the notebook's 3D surface: Pore fraction over (log10(Pi), log10(PR)).

    A per-material-coloured scatter of the experimental points is overlaid
    on a fitted 2D logistic-sigmoid surface  ``1 / (1 + exp(-(a·u + b·v + c)))``
    with ``u = log10(Pi)``, ``v = log10(PR)``.  This is the 3D generalisation
    of the 1D logistic collapse shown in panel 1 of
    :func:`plot_results`.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)
    from scipy.optimize import curve_fit

    os.makedirs(output_dir, exist_ok=True)
    u = np.log10(np.maximum(Pi, 1e-30))
    v = np.log10(np.maximum(PR, 1e-30))

    def _sigmoid2d(UV, a, b, c):
        uu, vv = UV
        return 1.0 / (1.0 + np.exp(-(a * uu + b * vv + c)))

    try:
        popt, _ = curve_fit(_sigmoid2d, (u, v), y,
                            p0=[2.0, 0.5, -2.0], maxfev=20000)
        yhat = _sigmoid2d((u, v), *popt)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        fit_ok = True
    except Exception:
        popt, r2, fit_ok = (0, 0, 0), float("nan"), False

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("LPBF Porosity — 3D Collapse on (log₁₀Π, log₁₀(P_recoil/P_Laplace))",
                 fontsize=13, fontweight="bold")

    # Fitted surface
    if fit_ok:
        u_grid = np.linspace(u.min(), u.max(), 40)
        v_grid = np.linspace(v.min(), v.max(), 40)
        U, V = np.meshgrid(u_grid, v_grid)
        Z = _sigmoid2d((U.ravel(), V.ravel()), *popt).reshape(U.shape)
        ax.plot_surface(U, V, Z, cmap="viridis", alpha=0.45,
                        linewidth=0, antialiased=True, edgecolor="none")

    # Scatter, coloured per material
    uniq_mats = sorted(set(materials.tolist()))
    palette = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
               "#a65628", "#f781bf"]
    for i, mat in enumerate(uniq_mats):
        m = (materials == mat)
        ax.scatter(u[m], v[m], y[m],
                   c=palette[i % len(palette)], s=28,
                   label=f"{mat} (n={m.sum()})",
                   edgecolors="black", linewidth=0.3, depthshade=True)

    ax.set_xlabel(r"$\log_{10}\Pi$ — normalised enthalpy", fontsize=10, labelpad=6)
    ax.set_ylabel(r"$\log_{10}(P_{recoil}/P_{Laplace})$", fontsize=10, labelpad=6)
    ax.set_zlabel("Pore fraction", fontsize=10, labelpad=4)
    if fit_ok:
        a, b, c = popt
        ax.set_title(
            f"Logistic surface   σ(a·log₁₀Π + b·log₁₀PR + c)   "
            f"a={a:+.2f}, b={b:+.2f}, c={c:+.2f}   R² = {r2:.3f}",
            fontsize=11,
        )
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.view_init(elev=22, azim=-58)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(output_dir, "lpbf_3d_surface.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"3D surface figure saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Discover symmetry in LPBF porosity data")
    parser.add_argument("--data", default="dataset_lpbf.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--output-dir", default="output_lpbf_porosity_symmetry")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[64, 32],
                        help="Hidden layer widths for the multilayer encoder "
                             "(default: 64 32)")
    parser.add_argument("--no-pi-only", action="store_true",
                        help="Disable the default pi-only mode: feed [X, X², log|X|, Pi] "
                             "to the Step 2 encoder instead of Pi groups alone.")
    parser.add_argument("--no-repo-da", action="store_true",
                        help="Use the inline dimensional-analysis implementation instead of "
                             "the repository's pydimension.data_preprocessing.DataPreprocessor "
                             "pipeline.")
    parser.add_argument("--log-normalize", action="store_true",
                        help="Geometric-mean centre each column before scaling. This makes "
                             "the scaling encoder's internal log(X) act as centred log-physical "
                             "coordinates, so discovered slopes map 1:1 onto power-law exponents.")
    args = parser.parse_args()
    args.pi_only = not args.no_pi_only

    X, y, Pi, PR, materials = load_data(args)
    results = run_pipeline(X, y, Pi, PR, materials, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_pi_candidates(X, y, results, args.output_dir)
    plot_results(X, y, results, args.output_dir)
    plot_3d_surface(Pi, PR, y, materials, args.output_dir)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    print(f"  Symmetry: {sym_type}")
    print(f"  Generators: {len(results['generators'])}")
    if sym_type == "scaling":
        print(f"  These generators show how P, V, A, rho, k, Lv, dT, γ, Tb can be")
        print(f"  simultaneously rescaled while preserving the normalised enthalpy Pi")
        print(f"  — and therefore the LPBF pore fraction.")
    if args.pi_only:
        print(f"  Step 2 (latent dim) used only the dimensionless Pi groups.")
        print(f"  Step 3 (symmetry type) used raw physical X for valid log/X²/X transforms.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
