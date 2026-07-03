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

plt.rcParams.update({
    "font.size":             17,
    "axes.titlesize":        20,
    "axes.labelsize":        19,
    "xtick.labelsize":       17,
    "ytick.labelsize":       17,
    "legend.fontsize":       17,
    "legend.title_fontsize": 18,
    "figure.titlesize":      24,
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

# Required: the repository's DataPreprocessor for dimensional analysis.
# This is the only Pi-discovery path the script supports — install
# pydimension (and its `seaborn` dependency) if the import below fails.
# The pydimension package lives at the repo root; walk up until we find it.
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

# Prevent silent multiprocessing crashes on Windows
import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

VARIABLE_NAMES = ["P", "V", "A", "rho", "k", "Lv", "dT", "gamma", "Tb",
                  "Tm_minus_T0"]
VARIABLE_UNITS = ["W", "m/s", "-", "kg/m³", "W/(m·K)", "J/kg", "K", "N/m", "K",
                  "K"]

# Dimension matrix, rows = (Mass, Length, Time, Temperature), cols = VARIABLE_NAMES.
#   P            [W]       = kg · m² · s⁻³          →  ( 1,  2, -3,  0)
#   V            [m/s]                               →  ( 0,  1, -1,  0)
#   A            [-]       (dimensionless)           →  ( 0,  0,  0,  0)
#   rho          [kg/m³]                              →  ( 1, -3,  0,  0)
#   k            [W/(m·K)] = kg · m · s⁻³ · K⁻¹       →  ( 1,  1, -3, -1)
#   Lv           [J/kg]    = m² · s⁻²                 →  ( 0,  2, -2,  0)
#   dT           [K]       (= Tb − Tm; the boil-melt superheat)  →  ( 0,  0,  0,  1)
#   gamma        [N/m]     = kg · s⁻²                 →  ( 1,  0, -2,  0)
#   Tb           [K]                                  →  ( 0,  0,  0,  1)
#   Tm_minus_T0  [K]       (the missing factor of Pe_vap)
DIMENSION_MATRIX = np.array([
    # P  V  A  rho  k  Lv  dT  gamma  Tb  Tm-T0
    [ 1, 0, 0,  1,  1,  0,  0,   1,   0,    0],   # Mass
    [ 2, 1, 0, -3,  1,  2,  0,   0,   0,    0],   # Length
    [-3,-1, 0,  0, -3, -2,  0,  -2,   0,    0],   # Time
    [ 0, 0, 0,  0, -1,  0,  1,   0,   1,    1],   # Temperature
], dtype=float)
DIMENSION_NAMES = ["Mass", "Length", "Time", "Temperature"]

# Pi (normalised enthalpy) known exponents, now over 10 variables.
# (Lv·rho·A·P·V) / (k^2·dT^2) — γ, Tb and Tm-T0 have zero exponent here.
#                              P    V    A    rho    k    Lv   dT   γ    Tb  Tm-T0
KNOWN_PI_EXPONENTS = np.array([1.0, 1.0, 1.0, 1.0, -2.0, 1.0, -2.0, 0.0, 0.0, 0.0])

# ──────────────────────────────────────────────────────────────────────────────
# Per-material thermophysical properties used to evaluate Pe_vap and Pr
# per row (and to supply the gamma, Tb columns missing from the CSV).
# ──────────────────────────────────────────────────────────────────────────────
PRESSURE_PROPS = {
    # T_boil [K], gamma [N/m], Tm [K] (melting point),
    # eta [Pa·s] (dynamic viscosity of melt near liquidus),
    # Cp  [J/(kg·K)] (specific heat of melt near liquidus)
    "Ti64":   dict(Tb=3560.0, gamma=1.65,
                   Tm=1923.0, eta=3.25e-3, Cp=700.0),
    "SS304":  dict(Tb=3090.0, gamma=1.80,
                   Tm=1700.0, eta=6.50e-3, Cp=750.0),
    "Al2024": dict(Tb=2792.0, gamma=0.90,
                   Tm=916.0,  eta=1.30e-3, Cp=1180.0),
    "Al6061": dict(Tb=2792.0, gamma=0.90,
                   Tm=925.0,  eta=1.30e-3, Cp=1180.0),
    "Cu":     dict(Tb=2835.0, gamma=1.30,
                   Tm=1358.0, eta=4.00e-3, Cp=510.0),
}
T_AMBIENT = 298.0   # K   (room temperature, used in Pe_vap)


def compute_pi(X: np.ndarray) -> np.ndarray:
    """Compute the notebook's normalised-enthalpy Pi from the 9-column X."""
    P, V, A, rho, k, Lv, dT = (X[:, i] for i in range(7))
    return (Lv * rho * A * P * V) / (k ** 2 * dT ** 2)


def compute_pe_vap(X: np.ndarray, Tm: np.ndarray,
                   T_ambient: float = T_AMBIENT) -> np.ndarray:
    """Vaporisation Peclet number per row (manuscript formula).

        Pe_vap = (Lv · rho · A · P · V) / (k^2 · (Tb − Tm) · (Tm − T0))

    Inputs: X columns = (P, V, A, rho, k, Lv, dT, gamma, Tb) where
    dT = (Tb − Tm) is the boil-melt superheat already in the dataset.
    Tm is supplied per row (material-dependent) and T0 defaults to 298 K.
    """
    P, V, A, rho, k, Lv, dT = (X[:, i] for i in range(7))
    return (Lv * rho * A * P * V) / (k ** 2 * dT * (Tm - T_ambient))


def compute_prandtl(eta: np.ndarray, Cp: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Thermal Prandtl number Pr = η · Cp / k (per row)."""
    return eta * Cp / k


# ──────────────────────────────────────────────────────────────────────────────
# Dimensional analysis — Stage-0 reduction to dimensionless candidates
# ──────────────────────────────────────────────────────────────────────────────
# Pi-basis discovery is always done via
# pydimension.data_preprocessing.DataPreprocessor (see
# run_repo_dimensional_analysis).  No inline fallback path exists.


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
    """Copy the dataset and append the extra per-row columns required for
    the 10-variable analysis: γ, Tb, Tm_minus_T0.  ((Tb − Tm) is already
    in the dataset as dT and is not duplicated.)

    DataPreprocessor reads variables straight from CSV columns, so any
    quantity that appears in VARIABLE_NAMES must exist as a column here.
    """
    import csv as _csv
    rows = []
    with open(src_path, "r") as f:
        rdr = _csv.DictReader(f)
        for r in rdr:
            mat = (r.get("source") or "").strip()
            if mat in PRESSURE_PROPS:
                props = PRESSURE_PROPS[mat]
                r["gamma"]       = props["gamma"]
                r["Tb"]          = props["Tb"]
                r["Tm_minus_T0"] = props["Tm"] - T_AMBIENT
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
    per-material γ (surface tension), Tb (boiling temperature), Tm
    (melting point), η (viscosity), and Cp (specific heat) are looked up
    from :data:`PRESSURE_PROPS` using the ``source`` column, so X is
    returned with 9 columns matching :data:`VARIABLE_NAMES`.
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

    X_list, y_list, mat_list = [], [], []
    Tm_list, eta_list, Cp_list = [], [], []
    for r in rows:
        try:
            mat = r[source_col].strip()
            if mat not in PRESSURE_PROPS:
                continue
            props = PRESSURE_PROPS[mat]
            row_vals = [float(r[c]) for c in input_cols]
            # Columns 8..9:  gamma, Tb, Tm_minus_T0.  (Tb − Tm is already
            # in the dataset as dT, so we don't duplicate it.)
            Tm_minus_T0 = props["Tm"] - T_AMBIENT
            row_vals.extend([props["gamma"], props["Tb"], Tm_minus_T0])
            X_list.append(row_vals)
            y_list.append(float(r[output_col]))
            mat_list.append(mat)
            Tm_list.append(props["Tm"])
            eta_list.append(props["eta"])
            Cp_list.append(props["Cp"])
        except (ValueError, IndexError):
            continue
    X = np.array(X_list)
    y = np.array(y_list)
    Tm  = np.array(Tm_list)
    eta = np.array(eta_list)
    Cp  = np.array(Cp_list)
    materials = np.array(mat_list)
    print(f"  Loaded: {CSV_VARS + ['gamma (from source)', 'Tb (from source)']} -> "
          f"{header[output_col].strip()}")
    print(f"  Unique materials: {sorted(set(mat_list))}")
    return {"X": X, "y": y, "materials": materials,
            "Tm": Tm, "eta": eta, "Cp": Cp}


def load_data(args):
    """Load data from CSV, compute Pi, Pe_vap, Pr.

    Returns
    -------
    X          : (n, 9)      physical inputs (P, V, A, rho, k, Lv, dT, γ, Tb)
    y          : (n,)        pore fraction, clipped to [0, 1]
    Pi         : (n,)        notebook's normalised enthalpy (reference)
    Pe_vap     : (n,)        vaporisation Péclet (manuscript formula)
    Pr_thermal : (n,)        thermal Prandtl η·Cp/k
    materials  : (n,)        material name per row, for plot colouring
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
    Tm        = data["Tm"]
    eta       = data["eta"]
    Cp        = data["Cp"]

    Pi = compute_pi(X)
    # Manuscript formulas (T_b - T_m)(T_m - T_0) and η·Cp/k
    Pe_vap     = compute_pe_vap(X, Tm)
    Pr_thermal = compute_prandtl(eta, Cp, X[:, 4])   # X[:,4] = k

    mask = (np.isfinite(Pi) & (Pi > 0)
            & np.isfinite(Pe_vap) & (Pe_vap > 0)
            & np.isfinite(Pr_thermal) & (Pr_thermal > 0))
    dropped = (~mask).sum()
    if dropped:
        print(f"  Dropping {dropped} rows with non-positive Pi/Pe_vap/Pr")
        X, y, Pi, Pe_vap, Pr_thermal, materials = (
            X[mask], y[mask], Pi[mask],
            Pe_vap[mask], Pr_thermal[mask], materials[mask],
        )

    y = np.clip(y, 0.0, 1.0)

    print(f"  Samples: {X.shape[0]}")
    print(f"  Pi range:     [{Pi.min():.4g}, {Pi.max():.4g}]")
    print(f"  Pe_vap range: [{Pe_vap.min():.4g}, {Pe_vap.max():.4g}]   "
          f"(Lv·ρ·A·P·V)/(k²·(Tb-Tm)·(Tm-T0))")
    print(f"  Pr range:     [{Pr_thermal.min():.4g}, {Pr_thermal.max():.4g}]   "
          f"(η·Cp/k, thermal Prandtl)")
    print(f"  Pore range:   [{y.min():.4f}, {y.max():.4f}]")
    print()
    return X, y, Pi, Pe_vap, Pr_thermal, materials


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 variant: concatenated-bottleneck identification
# ──────────────────────────────────────────────────────────────────────────────

def identify_symmetry_with_bottleneck_extras(
    X: np.ndarray,
    y: np.ndarray,
    bottleneck_extras: np.ndarray,
    n_latent: int,
    n_epochs: int = 1500,
    n_restarts: int = 3,
    seed: int = 0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    hidden_dim: int = 64,
    val_fraction: float = 0.2,
    device: str = "auto",
) -> dict:
    """Variant of identify_symmetry with extras concatenated post-encoder.

    For each symmetry class s, the encoder still applies the single linear
    layer  z_s = W_s · ϕ_s(X_enc)  to the raw physical input X_enc only.
    The decoder, however, sees the concatenated vector
        [z_s, bottleneck_extras]   (dim = n_latent + bottleneck_extras.shape[1])
    so the Pi quantities supplied in ``bottleneck_extras`` are given to the
    decoder for free, bypassing ϕ_s.
    """
    from symmetry_discovery.encoders import SymmetryEncoder
    import torch.nn as nn

    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    torch.manual_seed(seed)
    np.random.seed(seed)

    n_samples, n_inputs = X.shape
    n_extra = bottleneck_extras.shape[1] if bottleneck_extras.size else 0
    dec_in  = n_latent + n_extra

    n_val   = int(n_samples * val_fraction)
    idx     = np.random.permutation(n_samples)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    X_tr_t   = torch.tensor(X[tr_idx],   dtype=torch.float32).to(dev)
    y_tr_t   = torch.tensor(y[tr_idx],   dtype=torch.float32).unsqueeze(1).to(dev)
    X_val_t  = torch.tensor(X[val_idx],  dtype=torch.float32).to(dev)
    y_val_np = y[val_idx]
    if n_extra:
        E_tr_t  = torch.tensor(bottleneck_extras[tr_idx],  dtype=torch.float32).to(dev)
        E_val_t = torch.tensor(bottleneck_extras[val_idx], dtype=torch.float32).to(dev)
    else:
        E_tr_t  = torch.empty((X_tr_t.shape[0],  0), device=dev)
        E_val_t = torch.empty((X_val_t.shape[0], 0), device=dev)

    def _make_dec():
        return nn.Sequential(
            nn.Linear(dec_in, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    sym_types = ("translational", "rotational", "scaling")
    best_losses, best_encs, best_decs = {}, {}, {}

    for sym_type in sym_types:
        best_loss, best_enc, best_dec = np.inf, None, None
        for restart in range(n_restarts):
            torch.manual_seed(seed + hash(sym_type) % 1000 + restart * 37)
            enc = SymmetryEncoder(sym_type, n_inputs, n_latent).to(dev)
            dec = _make_dec().to(dev)
            opt = torch.optim.Adam(
                list(enc.parameters()) + list(dec.parameters()),
                lr=lr, weight_decay=weight_decay,
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=n_epochs, eta_min=lr * 0.01
            )
            loss_fn = nn.MSELoss()

            enc.train(); dec.train()
            n_tr = X_tr_t.shape[0]
            for _ in range(n_epochs):
                perm = torch.randperm(n_tr, device=dev)
                for start in range(0, n_tr, batch_size):
                    b = perm[start:start + batch_size]
                    opt.zero_grad()
                    z = enc(X_tr_t[b])
                    bottleneck = torch.cat([z, E_tr_t[b]], dim=1) if n_extra else z
                    loss_fn(dec(bottleneck), y_tr_t[b]).backward()
                    opt.step()
                sched.step()

            enc.eval(); dec.eval()
            with torch.no_grad():
                z_v = enc(X_val_t)
                bv  = torch.cat([z_v, E_val_t], dim=1) if n_extra else z_v
                pred = dec(bv).squeeze(1).cpu().numpy()
            val_loss = float(np.mean((y_val_np - pred) ** 2))
            if val_loss < best_loss:
                best_loss, best_enc, best_dec = val_loss, enc, dec

        best_losses[sym_type] = best_loss
        best_encs[sym_type]   = best_enc
        best_decs[sym_type]   = best_dec

    winner = min(best_losses, key=lambda t: best_losses[t])
    return {
        "symmetry_type": winner,
        "coefficients":  best_encs[winner].coefficients,
        "losses":        best_losses,
        "encoders":      best_encs,
        "decoders":      best_decs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(X, y, Pi, Pe_vap, Pr_thermal, materials, args):
    """Run Stage1 symmetry discovery on the LPBF physical variables."""
    results = {"Pi": Pi, "Pe_vap": Pe_vap, "Pr_thermal": Pr_thermal,
               "X_raw": X, "materials": materials}

    # --- Stage 0: Dimensional analysis → reduced Pi candidates ---
    print("=" * 60)
    print("Step 0: Dimensional analysis (Buckingham-Pi reduction)")
    print("=" * 60)
    print(f"  Dimension matrix shape: {DIMENSION_MATRIX.shape}  "
          f"(rows = {DIMENSION_NAMES}, cols = {VARIABLE_NAMES})")
    rank = int(np.linalg.matrix_rank(DIMENSION_MATRIX))
    print(f"  Rank: {rank}   Expected Pi groups: {DIMENSION_MATRIX.shape[1] - rank}")

    # Always use the repo's DataPreprocessor pipeline.  We hand it the
    # dataset CSV (enriched with γ, Tb, Tm_minus_T0 columns) and an
    # explicit dimension-matrix CSV so the result comes from pydimension's
    # null-space + SymPy primitive-integer reduction.
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
    # Verify the known normalised-enthalpy Pi lies in the null-space span.
    coords, *_ = np.linalg.lstsq(pi_basis, KNOWN_PI_EXPONENTS, rcond=None)
    recon = pi_basis @ coords
    ref_n = KNOWN_PI_EXPONENTS / (np.linalg.norm(KNOWN_PI_EXPONENTS) + 1e-12)
    recon_cos = float(np.dot(recon, ref_n) / (np.linalg.norm(recon) + 1e-12))
    print(f"  Known Pi exponents projected onto null-space basis: cos = {recon_cos:+.4f}  "
          f"(±1 means the notebook Pi lies in the Pi-group span)")
    pi_features = compute_pi_features(X, pi_basis)
    pi_feature_names = [f"Pi{i+1} (DA)" for i in range(pi_basis.shape[1])]

    # Append the two KNOWN dimensionless numbers from the keyhole-transition
    # literature as extra Pi features: the vaporisation Péclet Pe_vap
    # (normalised enthalpy) and the thermal Prandtl Pr.  Same treatment as
    # the DA groups: log10 then min-max to [0, 1].
    known_pi_vals = np.column_stack([Pe_vap, Pr_thermal])
    log_known = np.log10(np.maximum(known_pi_vals, 1e-30))
    mn_k = log_known.min(axis=0, keepdims=True)
    mx_k = log_known.max(axis=0, keepdims=True)
    rng_k = np.where(mx_k - mn_k > 1e-12, mx_k - mn_k, 1.0)
    known_features = (log_known - mn_k) / rng_k
    pi_features = np.hstack([pi_features, known_features])
    pi_feature_names += ["Pe_vap (known)", "Pr (known)"]

    results["pi_basis"] = pi_basis
    results["pi_features"] = pi_features
    results["pi_feature_names"] = pi_feature_names
    print(f"  Reduced candidates (pi_features) shape: {pi_features.shape}  "
          f"range: [{pi_features.min():.3f}, {pi_features.max():.3f}]")
    print(f"  Feature list: {pi_feature_names}")
    print(f"  (Pe_vap and Pr appended as known extra Pi features)")

    # Geometric-mean-centred Pi VALUES for Step 3 (DA groups + Pe_vap + Pr):
    # a purely multiplicative rescaling, so the scaling encoder's internal
    # log sees centred log-Pi coordinates and generators live in Pi space.
    X_pos = np.maximum(X, 1e-30)
    log10_all_pi = np.hstack([np.log10(X_pos) @ pi_basis,
                              np.log10(np.maximum(known_pi_vals, 1e-30))])
    log10_all_pi = log10_all_pi - log10_all_pi.mean(axis=0, keepdims=True)
    pi_centred = 10.0 ** log10_all_pi
    pi_names_step3 = [f"Pi{i+1}" for i in range(pi_basis.shape[1])] + ["Pe_vap", "Pr"]
    results["pi_centred"] = pi_centred
    results["pi_names_step3"] = pi_names_step3
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

    # --- Diagnostic: do the discovered latent dims encode Pe_vap and Pr? ---
    # If a manuscript-defined Pi (Pe_vap or thermal Prandtl) is *not* a linear
    # combination of the trained latent coordinates z, augment Step 3's input
    # with it so the single-layer symmetry encoder can pick it up.
    print("=" * 60)
    print("Step 2b: Checking whether z encodes Pe_vap and Pr")
    print("=" * 60)
    sys.stdout.flush()
    best_enc = res_latent["best_encoder"]
    with torch.no_grad():
        z_all = best_enc(
            torch.tensor(X_norm_step2, dtype=torch.float32)
        ).cpu().numpy()
    from sklearn.linear_model import LinearRegression
    log_pe = np.log10(Pe_vap)
    log_pr = np.log10(Pr_thermal)
    r2_pe = LinearRegression().fit(z_all, log_pe).score(z_all, log_pe)
    r2_pr = LinearRegression().fit(z_all, log_pr).score(z_all, log_pr)
    R2_DISCOVERED = 0.80
    print(f"  z (shape {z_all.shape}) -> log10(Pe_vap): R2 = {r2_pe:.4f}  "
          f"{'(discovered)' if r2_pe >= R2_DISCOVERED else '(NOT in latent span)'}")
    print(f"  z (shape {z_all.shape}) -> log10(Pr):     R2 = {r2_pr:.4f}  "
          f"{'(discovered)' if r2_pr >= R2_DISCOVERED else '(NOT in latent span)'}")
    results["z_step2"]   = z_all
    results["r2_pe_vap"] = r2_pe
    results["r2_pr"]     = r2_pr

    # Pe_vap and Pr are now first-class Step 2 AND Step 3 encoder inputs
    # (appended to the Pi feature set above), so no bottleneck injection is
    # needed in pi-only mode.  The diagnostic above stays as a report of
    # whether the latent z actually absorbed them.
    if pi_only:
        names_step3 = list(pi_names_step3)
        print(f"  → Pe_vap and Pr are Step 2/3 encoder inputs (pi-only mode); "
              f"no bottleneck injection.")
    else:
        names_step3 = list(VARIABLE_NAMES)
    results["feature_names_step3"] = names_step3
    results["bottleneck_extras_names"] = []
    print()

    # --- Identify symmetry type ---
    print("=" * 60)
    print("Step 3: Identifying symmetry type")
    print("=" * 60)
    sys.stdout.flush()
    if pi_only:
        print(f"  Encoder input: {pi_centred.shape[1]} geometric-mean-centred "
              f"Pi values ({names_step3})")
        print(f"  Centring is purely multiplicative → the scaling encoder's "
              f"internal log sees centred log-Pi coordinates;")
        print(f"  generators live in dimensionless Pi space.")
        X_step3 = pi_centred
    else:
        X_step3 = X_norm_raw
    results["X_step3"] = X_step3
    res_sym = identify_symmetry(
        X_step3, y_norm, n_latent=n_latent,
        n_epochs=args.sym_epochs,
        n_restarts=args.n_restarts,
        seed=args.seed,
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
    # In pi-only mode Step 3 runs on the centred Pi values, so W is a
    # direction in log-Pi space.  Two known references there:
    #   (a) the notebook normalised-enthalpy Pi expressed in the DA basis
    #       (coords from lstsq, zero-padded over Pe_vap / Pr), and
    #   (b) the pure Pe_vap axis — Pe_vap IS the known collapse variable.
    W = winner_encoder.weight_matrix  # (n_latent, n_inputs)
    names_for_W = names_step3
    print("=" * 60)
    print("  Winning encoder weight vector(s)")
    print("=" * 60)
    name_w = max(7, max(len(n) for n in names_for_W))
    if pi_only:
        known_coords, *_ = np.linalg.lstsq(pi_basis, KNOWN_PI_EXPONENTS, rcond=None)
        refs = {
            "known-Pi (DA coords)": np.concatenate([known_coords, [0.0, 0.0]]),
            "pure Pe_vap axis":     np.eye(len(names_for_W))[len(names_for_W) - 2],
        }
    else:
        ref = np.zeros(len(names_for_W))
        ref[: len(KNOWN_PI_EXPONENTS)] = KNOWN_PI_EXPONENTS
        refs = {"known-Pi-exponents": ref}
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
        for label, ref in refs.items():
            ref_n = ref / (np.linalg.norm(ref) + 1e-12)
            cos = float(np.dot(row_n, ref_n))
            print(f"  cos<row, {label}> = {cos:+.4f}  "
                  f"(±1 means perfect alignment)")
    print()

    # --- Interpret generators physically ---
    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "scaling" and generators:
        print(f"  Each generator is a direction in log-Pi space along which the")
        print(f"  pore fraction is preserved: simultaneously rescaling the")
        print(f"  dimensionless groups along g keeps the LPBF physics invariant.\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = []
                for j, name in enumerate(names_step3):
                    if abs(g[j]) > 0.05:
                        parts.append(f"{name} x exp({g[j]:+.3f}*eps)")
                print(f"  Generator {i+1}:")
                print(f"    {', '.join(parts)}")
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
    abs_g = np.abs(g)
    dominant = np.argmax(abs_g)
    name = names[dominant]

    coupled = [(names[j], g[j]) for j in range(len(g))
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
                 fontweight="bold")

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
    for i in range(n_pi):
        for j in range(len(VARIABLE_NAMES)):
            v = pi_basis[j, i]
            if abs(v) > 1e-10:
                ax.text(j, i, f"{v:+.0f}" if abs(v - round(v)) < 1e-9 else f"{v:+.2f}",
                        ha="center", va="center",
                        color="white" if abs(v) > 0.6 * np.max(np.abs(pi_basis)) else "black",
                        fontsize=13)
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
                        va="top", ha="left", fontsize=13,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
            except Exception:
                pass
            ax.legend(fontsize=13, loc="lower right")
        expr = format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)
        ax.set_xlabel(f"log₁₀(Pi{i+1})\n{expr}", fontsize=14)
        ax.set_ylabel("Pore fraction")
        ax.set_title(f"Reduced candidate Pi{i+1}")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "lpbf_pi_candidates.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Pi candidates figure saved to {out_path}")


def plot_discovered_law_and_generators(y, results, output_dir):
    """Visualize the discovered coefficients and generators in Pi space.

    Four panels (requires pi-only mode and a scaling winner):
      A. Coefficient heatmap: the k* winning encoder rows (L2-normalised)
         stacked above the two known references — the notebook Pi expressed
         in DA coordinates, and the pure Pe_vap axis.
      B. Latent collapse: for k* = 2 a scatter of (z1, z2) colored by pore
         fraction; for k* = 1 a scatter of pore fraction vs z1.
      C. Heatmap of the null-space generators (rows) × Pi features (cols).
      D. Orbit invariance: Pi(ε) = Pi0 · exp(ε·g).  The known normalised-
         enthalpy Pi moves by exp(ε · c·g) exactly (c = its DA coordinates,
         zero-padded over Pe_vap/Pr); the latent z stays flat by null(W).
    """
    os.makedirs(output_dir, exist_ok=True)
    winner_encoder = results["winner_encoder"]
    generators     = results["generators"]
    W              = winner_encoder.weight_matrix          # (k*, n_pi)
    pi_centred     = results["pi_centred"]
    pi_names       = results["pi_names_step3"]
    pi_basis       = results["pi_basis"]
    n_pi           = pi_centred.shape[1]
    n_lat          = W.shape[0]
    n_gen          = len(generators)

    known_coords, *_ = np.linalg.lstsq(pi_basis, KNOWN_PI_EXPONENTS, rcond=None)
    c_pad = np.concatenate([known_coords, [0.0, 0.0]])     # over the 8 features
    c_dir = c_pad / (np.linalg.norm(c_pad) + 1e-12)
    pe_axis = np.zeros(n_pi)
    pe_axis[n_pi - 2] = 1.0

    W_dirs = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)

    # Latent coordinates (encoder-equivalent, incl. the 0.1 clamp)
    log_pi = np.log(np.clip(np.abs(pi_centred), 0.1, None))
    Z = log_pi @ W.T                                       # (n, k*)

    # Orbits from the geometric centre (centred Pi ⇒ start point = 1s)
    eps_grid = np.linspace(-0.5, 0.5, 41)
    orbit_pi_known = np.zeros((n_gen, eps_grid.size))
    orbit_dz_max   = np.zeros((n_gen, eps_grid.size))
    for k, g in enumerate(generators):
        orbit_pi_known[k] = np.exp(eps_grid * float(c_pad @ g))
        Pi_orbit = np.exp(np.outer(eps_grid, g))
        Zo = np.log(np.clip(Pi_orbit, 0.1, None)) @ W.T    # (n_eps, k*)
        orbit_dz_max[k] = np.abs(Zo).max(axis=1)           # z(0) = 0 at centre

    fig = plt.figure(figsize=(17, 13))
    gs  = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.45)
    fig.suptitle("LPBF Porosity — Discovered Coefficients & Generators in Pi Space",
                 fontweight="bold")

    # ── Panel A: coefficient heatmap, W rows vs known references ────────────
    ax = fig.add_subplot(gs[0, 0])
    rows   = [W_dirs[i] for i in range(n_lat)] + [c_dir, pe_axis]
    labels = [f"W row {i+1}" for i in range(n_lat)] + ["known Pi (DA)", "Pe_vap axis"]
    M = np.stack(rows, axis=0)
    vmax = float(np.max(np.abs(M))) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(n_pi))
    ax.set_xticklabels(pi_names, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels)
    ax.axhline(n_lat - 0.5, color="black", lw=1.5)
    for i in range(len(rows)):
        for j in range(n_pi):
            v = M[i, j]
            if abs(v) > 0.05:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 * vmax else "black",
                        fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cos_txt = "   ".join(
        f"cos(W{i+1},knownPi)={float(W_dirs[i] @ c_dir):+.2f}"
        for i in range(n_lat))
    ax.set_title(f"Discovered coefficients (L2-n) vs known references\n{cos_txt}",
                 fontsize=18)

    # ── Panel B: latent collapse ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    if n_lat >= 2:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="Blues", s=30,
                        edgecolors="#666666", linewidths=0.4)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="pore fraction")
        ax.set_xlabel("z₁ = W₁ · log(Pi_centred)")
        ax.set_ylabel("z₂ = W₂ · log(Pi_centred)")
        ax.set_title("Pore fraction over the 2-D discovered latent")
    else:
        ax.scatter(Z[:, 0], y, c="#4C72B0", s=25, alpha=0.75, edgecolors="none")
        ax.set_xlabel("z = W · log(Pi_centred)")
        ax.set_ylabel("pore fraction")
        ax.set_title("Pore fraction vs the discovered latent")

    # ── Panel C: generator heatmap ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if n_gen > 0:
        G = np.stack([np.asarray(g).ravel() for g in generators], axis=0)
        vmax = float(np.max(np.abs(G))) or 1.0
        im = ax.imshow(G, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n_pi))
        ax.set_xticklabels(pi_names, rotation=30, ha="right")
        ax.set_yticks(range(n_gen))
        ax.set_yticklabels([f"g{i+1}" for i in range(n_gen)])
        for i in range(n_gen):
            for j in range(n_pi):
                v = G[i, j]
                if abs(v) > 0.05:
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            color="white" if abs(v) > 0.6 * vmax else "black",
                            fontsize=13)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{n_gen} generators of pore-fraction invariance "
                     f"(rows = null(W))")
    else:
        ax.set_axis_off()
        ax.set_title("No generators")

    # ── Panel D: invariance along orbits ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    cmap = plt.get_cmap("tab10")
    for k in range(n_gen):
        ax.plot(eps_grid, orbit_pi_known[k], "-", color=cmap(k % 10),
                lw=1.8, label=f"known Pi, g{k+1}")
    if n_gen > 0:
        ax.plot(eps_grid, 1.0 + orbit_dz_max.max(axis=0), "k--", lw=2.2,
                alpha=0.9, label="1 + max|Δz|")
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_xlabel("Orbit parameter  ε   (Pi → Pi · exp(ε·g))")
    ax.set_ylabel("ratio to ε=0")
    ax.set_title("Invariance check: known Pi along each orbit")
    ax.legend(loc="best", ncol=2, fontsize=13)
    dev = float(np.max(np.abs(orbit_pi_known - 1.0))) if n_gen else 0.0
    ax.text(0.03, 0.03,
            f"max |ΔPi/Pi|={dev:.2f} at |ε|=0.5\n"
            "z flat by null(W); known-Pi drift reflects W–Pi misalignment\n"
            "and the Pe_vap/DA-group collinearity (gauge directions)",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=13,
            color="#333333")

    # Footer: the actual Pi-group expressions behind the feature names.
    da_exprs = [f"{pi_names[i]} = "
                f"{format_pi_expression(pi_basis[:, i], VARIABLE_NAMES)}"
                for i in range(pi_basis.shape[1])]
    known_exprs = [
        "Pe_vap = (Lv·rho·A·P·V) / (k²·dT·(Tm−T0))",
        "Pr = η·Cp / k   (η, Cp per material)",
    ]
    n_half = (len(da_exprs) + 1) // 2
    fig.text(0.05, 0.055, "\n".join(da_exprs[:n_half]),
             ha="left", va="top", fontsize=14, color="#444444")
    fig.text(0.37, 0.055, "\n".join(da_exprs[n_half:]),
             ha="left", va="top", fontsize=14, color="#444444")
    fig.text(0.69, 0.055, "\n".join(known_exprs),
             ha="left", va="top", fontsize=14, color="#444444")

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    out_path = os.path.join(output_dir, "lpbf_discovered_law_generators.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Discovered-coefficients & generators figure saved to {out_path}")


def plot_results(X, y, results, output_dir):
    """Two-panel summary: symmetry type bar chart + latent-dimension R² curve."""
    os.makedirs(output_dir, exist_ok=True)
    winner_type = results["winner_type"]
    sym_res = results["symmetry"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("LPBF Porosity — Symmetry Discovery", fontweight="bold")

    # --- Panel 1: Symmetry type identification ---
    ax = axes[0]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Symmetry Type  (winner: {sym_res['symmetry_type']})")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom", fontsize=14)
    sorted_losses = sorted(losses)
    if len(sorted_losses) >= 2 and sorted_losses[0] > 0:
        gap = sorted_losses[1] / sorted_losses[0]
        ax.text(0.97, 0.97, f"Loss gap: {gap:.1f}\u00d7",
                ha="right", va="top", transform=ax.transAxes,
                fontsize=14, color="#333333")

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
    ax.set_xlabel("Latent dimension k")
    ax.set_ylabel("R\u00b2")
    ax.set_title("Latent Dimension Discovery")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.05)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "lpbf_porosity_symmetry_discovery.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


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
    parser.add_argument("--log-normalize", action="store_true",
                        help="Geometric-mean centre each column before scaling. This makes "
                             "the scaling encoder's internal log(X) act as centred log-physical "
                             "coordinates, so discovered slopes map 1:1 onto power-law exponents.")
    args = parser.parse_args()
    args.pi_only = not args.no_pi_only

    X, y, Pi, Pe_vap, Pr_thermal, materials = load_data(args)
    results = run_pipeline(X, y, Pi, Pe_vap, Pr_thermal, materials, args)

    print("=" * 60)
    print("Creating visualizations")
    print("=" * 60)
    plot_pi_candidates(X, y, results, args.output_dir)
    plot_results(X, y, results, args.output_dir)
    if args.pi_only and results["winner_type"] == "scaling":
        plot_discovered_law_and_generators(y, results, args.output_dir)
    else:
        print("Skipping Pi-space coefficients/generator figure "
              "(requires pi-only mode and a scaling winner).")

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    sym_type = results["symmetry"]["symmetry_type"]
    print(f"  Symmetry: {sym_type}")
    print(f"  Generators: {len(results['generators'])}")
    if sym_type == "scaling":
        print(f"  These generators show how the dimensionless Pi groups (incl. the")
        print(f"  known Pe_vap and Pr) can be simultaneously rescaled while")
        print(f"  preserving the LPBF pore fraction.")
    if args.pi_only:
        print(f"  Step 2 (latent dim) used the DA Pi groups + known Pe_vap, Pr.")
        print(f"  Step 3 (symmetry type) used geometric-mean-centred Pi values.")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
