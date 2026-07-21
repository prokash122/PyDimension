"""
Discover symmetry in concrete compressive strength data using a
dimensionless representation instead of raw kg/m³ inputs.

Preprocessing (derived from Yeh, Cem. Concr. Res. 28(12), 1998)
---------------------------------------------------------------
Each mass quantity carries the dimension [M L^-3], so simply dividing it
by the total binder mass b = cement + slag + fly ash yields a
dimensionless ratio -- the standard concrete-science normalization (the
water/binder ratio, SCM replacement fractions, aggregate/binder ratios).
No Buckingham-Pi bookkeeping is needed; the binder mass is just chosen as
the common reference:

    pi_1 = water / b                        (literal water/binder ratio)
    pi_2 = cement / b
    pi_3 = slag / b
    pi_4 = fly_ash / b
    pi_5 = superplasticizer / b
    pi_6 = coarse_aggregate / b
    pi_7 = fine_aggregate / b
    pi_8 = ln(t / 28 days)                  (dimensionless age)

All eight non-dimensional candidates are kept (the seven mass ratios plus
the log-age term). Note that the three binder fractions are linearly
dependent (cement/b + slag/b + fly_ash/b = 1), so the feature set is
rank-deficient by one; this puts the data on a 7-D hyperplane and adds one
trivial "stay on the simplex" direction to the translational null space,
on top of the physical strength-preserving generators.

Superplasticizer keeps its own ratio pi_4; it is NOT folded into the
water term, so w/b here is the literal water-to-binder ratio.

The target is the residual strength ratio against Yeh's regression
baseline (Table 6, average of random-split experiments R1-R4):

    sigma_ideal = a * (w/b)^b_exp * (c*ln(t) + d)     [MPa, t in days]
    a = 13.83, b_exp = -1.269, c = 0.268, d = 0.136

where w/b is the same literal water/binder ratio as pi_1. Because
sigma_ideal already carries the dominant w/b and age effects, the ML
pipeline only has to model the residual chemistry (SCM substitution,
superplasticizer, aggregates).

Usage
-----
    python discover_symmetry_dimensionless.py --data Concrete_Data.xls
"""

import sys
import os
import argparse
import traceback
import multiprocessing

import numpy as np

# Add Stage1 modules to path
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
    "font.size":             13,
    "axes.titlesize":        16,
    "axes.labelsize":        15,
    "xtick.labelsize":       15,
    "ytick.labelsize":       15,
    "legend.fontsize":       13,
    "legend.title_fontsize": 14,
    "figure.titlesize":      19,
})

try:
    from preprocessing.normalize import normalize_data
    from intrinsic_coordinate.discovery import discover_latent_dimension
    from symmetry_discovery.identification import identify_symmetry
    from symmetry_discovery.generators import extract_generators
except ImportError as e:
    print(f"ERROR: Could not import Stage1 modules: {e}")
    print(f"Copy preprocessing/, intrinsic_coordinate/, symmetry_discovery/ from")
    print(f"projects/20260912_Stage1_Prokash/ into the same directory as this script.")
    sys.exit(1)

import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

# Yeh (1998) Table 6 coefficients, averaged over random-split runs R1-R4.
YEH_A = 13.83       # MPa — strength scale at w/b = 1
YEH_B = -1.269      # w/b exponent
YEH_C = 0.268       # age log-law slope
YEH_D = 0.136       # age log-law intercept (t in days)
T_REF = 28.0        # days

PI_NAMES = [
    "w/b", "Cement/b", "Slag/b", "FlyAsh/b", "SP/b",
    "CoarseAgg/b", "FineAgg/b", "ln(t/28)",
]


def load_raw_data(path: str):
    """Load the 8 raw columns + strength from CSV/XLS."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xls", ".xlsx"):
        import pandas as pd
        df = pd.read_excel(path)
        X = df.iloc[:, :8].values.astype(float)
        y = df.iloc[:, 8].values.astype(float)
        print(f"  Loaded Excel: {df.columns.tolist()}")
        return X, y
    import csv
    with open(path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    X = np.array([[float(row[i]) for i in range(8)] for row in rows])
    y = np.array([float(row[8]) for row in rows])
    print(f"  Loaded CSV: {[h.strip() for h in header]}")
    return X, y


def make_dimensionless(X_raw, sigma):
    """Binder-referenced dimensionless features and Yeh-residual target.

    Every mass is divided by the total binder mass b = cement + slag +
    fly ash (no Buckingham-Pi machinery -- b is just the reference).
    Returns Pi (n, 7), y = sigma/sigma_ideal, and sigma_ideal.
    """
    cement, slag, flyash, water, sp, ca, fa, age = X_raw.T
    binder = cement + slag + flyash
    wb = water / binder                 # literal water/binder ratio (no SP)

    # All eight non-dimensional candidates: the seven mass ratios (each
    # mass / binder, including all three binder fractions) plus log-age.
    # The binder fractions sum to 1, so the set is rank-deficient by one.
    Pi = np.column_stack([
        wb,
        cement / binder,
        slag / binder,
        flyash / binder,
        sp / binder,
        ca / binder,
        fa / binder,
        np.log(age / T_REF),
    ])

    sigma_ideal = YEH_A * wb**YEH_B * (YEH_C * np.log(age) + YEH_D)
    y = sigma / sigma_ideal
    return Pi, y, sigma_ideal


def run_pipeline(Pi, y, args):
    results = {}

    print("=" * 60)
    print("Step 1: Normalizing dimensionless data")
    print("=" * 60)
    sys.stdout.flush()
    norm = normalize_data(Pi, y, method="standard")
    X_norm, y_norm = norm["X_normalized"], norm["y_normalized"]
    results["normalization"] = norm
    print(f"  Pi range after scaling: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
    print()

    print("=" * 60)
    print("Step 2: Discovering intrinsic latent dimension")
    print("=" * 60)
    sys.stdout.flush()
    enc_kwargs = {"raw_input": True}
    if getattr(args, "encoder_hidden", None):
        enc_kwargs["encoder_hidden_dims"] = args.encoder_hidden
    print(f"  Multi-layer encoder hidden dims: {args.encoder_hidden}")

    res_latent = discover_latent_dimension(
        X_norm, y_norm, max_latent=args.max_latent,
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        **enc_kwargs)
    results["latent"] = res_latent
    n_latent = res_latent["optimal_n_latent"]
    decoder = res_latent["best_decoder"]
    if getattr(args, "latent_dim", None):
        # Override the auto-selected k. The per-k reconstruction MSEs are
        # nearly tied on this residual, so the argmin is noise-sensitive;
        # pinning k keeps the generator count reproducible across runs.
        print(f"\n  Auto-selected latent dimension: {n_latent} "
              f"(overridden to {args.latent_dim} via --latent-dim)")
        n_latent = args.latent_dim
        decoder = res_latent["models_per_k"][n_latent].decoder
    print(f"\n  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        r2_tr = m.get("R2_train", float("nan"))
        print(f"    k={k}: R2_train={r2_tr:.4f}, R2_test={m['R2']:.4f}, MSE={m['MSE']:.6f}")
    print()

    print("=" * 60)
    print("Step 3: Identifying symmetry type")
    print("=" * 60)
    sys.stdout.flush()
    res_sym = identify_symmetry(
        X_norm, y_norm, n_latent=n_latent, decoder=decoder,
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

    print("=" * 60)
    print("Step 4: Extracting Lie-algebra generators")
    print("=" * 60)
    winner_type = res_sym["symmetry_type"]
    winner_encoder = res_sym["encoders"][winner_type]
    generators = extract_generators(winner_type, winner_encoder)
    results["generators"] = generators
    results["winner_type"] = winner_type
    results["W"] = winner_encoder.weight_matrix

    print(f"  Symmetry type: {winner_type}")
    print(f"  Generators: {len(generators)}")
    print()

    print("=" * 60)
    print("Step 5: Physical interpretation of generators")
    print("=" * 60)
    if winner_type == "translational" and generators:
        print("  Each generator g shifts the dimensionless mix ratios,")
        print("  pi -> pi + eps*g, without changing the strength residual y.\n")
        for i, g in enumerate(generators):
            parts = [f"{PI_NAMES[j]}: {g[j]:+.3f}"
                     for j in range(len(PI_NAMES)) if abs(g[j]) > 0.05]
            print(f"  Generator {i+1}: [{', '.join(parts)}]")
        print()
    elif winner_type == "scaling" and generators:
        print("  Each generator is a log-space direction: pi -> pi*exp(eps*s).\n")
        for i, g in enumerate(generators):
            if g.ndim == 1:
                parts = [f"{PI_NAMES[j]} x exp({g[j]:+.3f}*eps)"
                         for j in range(len(PI_NAMES)) if abs(g[j]) > 0.05]
                print(f"  Generator {i+1}: [{', '.join(parts)}]")
        print()
    elif winner_type == "rotational" and generators:
        for i, g in enumerate(generators):
            print(f"  Generator {i+1} (antisymmetric matrix):")
            print(f"    {np.round(g, 4)}")
        print()

    return results


def plot_results(sigma, sigma_ideal, y, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    norm = results["normalization"]
    sym_res = results["symmetry"]
    W = results["W"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("Concrete Strength — Symmetry Discovery on Dimensionless Data",
                 fontweight="bold")

    # Panel 1: Yeh baseline vs measured strength
    ax = axes[0]
    ax.scatter(sigma_ideal, sigma, c="#4C72B0", s=12, alpha=0.5, edgecolors="none")
    lim = [0, max(sigma.max(), sigma_ideal.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1.5, label="1:1")
    ss_res = np.sum((sigma - sigma_ideal) ** 2)
    ss_tot = np.sum((sigma - sigma.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    ax.set_xlabel(r"$\sigma_{ideal}$ (Yeh 1998 baseline, MPa)")
    ax.set_ylabel(r"Measured $\sigma_c$ (MPa)")
    ax.set_title(f"Yeh Baseline (R²={r2:.3f})")
    ax.legend()

    # Panel 2: learned latent embedding colored by residual
    ax = axes[1]
    X_norm = norm["X_normalized"]
    z = X_norm @ W.T
    if z.shape[1] == 1:
        z1 = z.ravel()
        ax.scatter(z1, y, c="#4C72B0", s=12, alpha=0.5, edgecolors="none")
        ax.set_xlabel("z = W·π (learned latent variable)")
        ax.set_ylabel(r"$y = \sigma/\sigma_{ideal}$")
        ax.set_title("Latent Variable vs Strength Residual")
    else:
        sc = ax.scatter(z[:, 0], z[:, 1], c=y, cmap="viridis", s=12, alpha=0.6)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r"$\sigma/\sigma_{ideal}$", fontsize=15)
        cbar.ax.tick_params(labelsize=15)
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        ax.set_title("Latent Variables (colored by residual)")

    # Panel 3: symmetry-type competition
    ax = axes[2]
    types = list(sym_res["losses"].keys())
    losses = [sym_res["losses"][t] for t in types]
    colors = ["#55A868" if t == sym_res["symmetry_type"] else "#DD8452" for t in types]
    bars = ax.bar(types, losses, color=colors, edgecolor="black", lw=1)
    ax.set_ylim(0, max(1.0, max(losses) * 1.15))
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Symmetry Type (winner: {sym_res['symmetry_type']})")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.4f}", ha="center", va="bottom")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = os.path.join(output_dir, "concrete_symmetry_dimensionless.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Symmetry discovery on dimensionless concrete data"
    )
    parser.add_argument("--data", default="Concrete_Data.xls")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-epochs", type=int, default=600)
    parser.add_argument("--sym-epochs", type=int, default=1500)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--max-latent", type=int, default=6,
                        help="Largest latent dimension to test (must be < 8 "
                             "so that translational generators remain)")
    parser.add_argument("--latent-dim", type=int, default=5,
                        help="Pin the latent dimension k. Default 5 (the value "
                             "the auto-search selects on GPU; on some CPUs the "
                             "near-tied MSEs make it pick 4). Set to 0 to let "
                             "the pipeline choose automatically.")
    parser.add_argument("--output-dir", default="output_concrete_dimensionless")
    parser.add_argument("--encoder-hidden", type=int, nargs="+", default=[64, 32])
    args = parser.parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        data_path = os.path.join(_here, os.path.basename(args.data))
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    print(f"Loading concrete data from {data_path}...")
    X_raw, sigma = load_raw_data(data_path)
    print(f"  Samples: {X_raw.shape[0]}, Raw features: {X_raw.shape[1]}")
    print(f"  Strength range: [{sigma.min():.1f}, {sigma.max():.1f}] MPa")
    print()

    print("=" * 60)
    print("Step 0: Binder-referenced non-dimensionalization")
    print("=" * 60)
    Pi, y, sigma_ideal = make_dimensionless(X_raw, sigma)
    print(f"  Features: {PI_NAMES}")
    print(f"  sigma_ideal = {YEH_A}*(w/b)^({YEH_B})*({YEH_C}*ln(t)+{YEH_D})  [MPa]")
    r2_base = 1 - np.sum((sigma - sigma_ideal) ** 2) / np.sum((sigma - sigma.mean()) ** 2)
    print(f"  Yeh baseline alone: R2 = {r2_base:.4f} on all {len(sigma)} rows")
    print(f"  Residual target y = sigma/sigma_ideal: "
          f"mean={y.mean():.4f}, std={y.std():.4f}")
    print()

    results = run_pipeline(Pi, y, args)

    os.makedirs(args.output_dir, exist_ok=True)
    scaler_y = results["normalization"]["scaler_y"]
    sym_losses = results["symmetry"]["losses"]
    np.savez(
        os.path.join(args.output_dir, "pipeline_artifacts.npz"),
        Pi=Pi, y=y, sigma=sigma, sigma_ideal=sigma_ideal,
        X_norm=results["normalization"]["X_normalized"],
        W=results["W"],
        generators=np.array(results["generators"]),
        pi_names=np.array(PI_NAMES),
        # standard-scaler of the residual target y, to invert decoder output
        y_mean=np.asarray(getattr(scaler_y, "mean_", 0.0), dtype=float),
        y_std=np.asarray(getattr(scaler_y, "std_", 1.0), dtype=float),
        # symmetry-type competition (for the standalone bar-chart figure)
        sym_types=np.array(list(sym_losses.keys())),
        sym_losses=np.array(list(sym_losses.values()), dtype=float),
        winner_type=np.array(results["winner_type"]),
    )
    # Save the GENUINE trained model: winning translational encoder + its
    # jointly trained decoder, so downstream plots run the real model end-to-end.
    winner = results["symmetry"]["symmetry_type"]
    import torch
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
    plot_results(sigma, sigma_ideal, y, results, args.output_dir)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Symmetry: {results['symmetry']['symmetry_type']}")
    print(f"  Generators: {len(results['generators'])}")
    print()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
