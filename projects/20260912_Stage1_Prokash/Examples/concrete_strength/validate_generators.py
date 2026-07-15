"""
Empirical validation of the discovered translational generators against
real (measured) data — no model prediction is used anywhere.

Logic
-----
The translational encoder z = W*pi (W: k x 7) splits standardized
pi-space into two orthogonal subspaces:

  * ACTIVE subspace  (row space of W, dim k):   moving here changes z,
    and hence the predicted strength residual.
  * SYMMETRY subspace (null space of W, dim 7-k): spanned by the
    generators; moving here should leave the strength residual
    unchanged.

If the generators are real physics and not an artifact of the network,
then for PAIRS OF ACTUAL MIXES in the dataset:

  * pairs separated (in standardized pi-space) almost purely along the
    SYMMETRY subspace should have nearly equal measured sigma/sigma_ideal;
  * pairs separated almost purely along the ACTIVE subspace should show
    systematically different measured sigma/sigma_ideal;
  * near-duplicate mixes (tiny separation) give the experimental
    repeatability noise floor that symmetry-aligned pairs should approach.

This uses only measured strengths on both sides of the comparison.

Usage
-----
    python validate_generators.py            # after discover_symmetry_dimensionless.py
    python validate_generators.py --align 0.9 --dmin 0.5 --dmax 2.5
"""

import os
import argparse

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
except (AttributeError, ImportError):
    pass
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  14,
    "axes.labelsize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

_here = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description="Validate discovered generators against measured mix pairs")
    parser.add_argument("--artifacts",
                        default="output_concrete_dimensionless/pipeline_artifacts.npz")
    parser.add_argument("--output-dir", default="output_concrete_dimensionless")
    parser.add_argument("--align", type=float, default=0.90,
                        help="Minimum |cos| alignment of a pair's separation "
                             "with the tested subspace/direction")
    parser.add_argument("--dmin", type=float, default=0.5,
                        help="Minimum pair separation (standardized pi units)")
    parser.add_argument("--dmax", type=float, default=2.5,
                        help="Maximum pair separation (standardized pi units)")
    parser.add_argument("--dup", type=float, default=0.05,
                        help="Separation below which a pair counts as a "
                             "near-duplicate (noise floor)")
    args = parser.parse_args()

    path = args.artifacts
    if not os.path.exists(path):
        path = os.path.join(_here, args.artifacts)
    data = np.load(path, allow_pickle=True)
    X = data["X_norm"]                 # standardized pi features (n, 7)
    y = data["y"]                      # measured sigma / sigma_ideal (n,)
    W = data["W"]                      # encoder weights (k, 7)
    gens = data["generators"]          # (7-k, 7), null-space basis
    pi_names = list(data["pi_names"])

    n, p = X.shape
    k = W.shape[0]
    print(f"Loaded {n} mixes, {p} features, latent dim k={k}, "
          f"{gens.shape[0]} generators")

    # Orthonormal bases of the two subspaces from the SVD of W.
    _, _, Vt = np.linalg.svd(W, full_matrices=True)
    row_basis = Vt[:k]                 # active subspace (k, 7)
    null_basis = Vt[k:]                # symmetry subspace (7-k, 7)
    resid = np.abs(gens @ row_basis.T).max()
    print(f"Generator/active-subspace overlap (should be ~0): {resid:.2e}")

    # All pairs.
    ii, jj = np.triu_indices(n, 1)
    D = X[ii] - X[jj]                          # (n_pairs, 7)
    dy = np.abs(y[ii] - y[jj])                 # measured residual difference
    d_row = np.linalg.norm(D @ row_basis.T, axis=1)
    d_null = np.linalg.norm(D @ null_basis.T, axis=1)
    d_tot = np.sqrt(d_row**2 + d_null**2)
    print(f"Total pairs: {len(dy)}")

    band = (d_tot >= args.dmin) & (d_tot <= args.dmax)
    frac_null = np.where(d_tot > 0, d_null / np.maximum(d_tot, 1e-12), 0.0)
    frac_row = np.where(d_tot > 0, d_row / np.maximum(d_tot, 1e-12), 0.0)

    sym_pairs = band & (frac_null >= args.align)     # move along symmetry
    act_pairs = band & (frac_row >= args.align)      # move along active dirs
    dup_pairs = d_tot <= args.dup                    # repeatability floor

    groups = {
        "near-duplicates\n(noise floor)": dup_pairs,
        "symmetry-aligned\npairs": sym_pairs,
        "random pairs\n(same |Δπ|)": band,
        "active-aligned\npairs": act_pairs,
    }
    print()
    print(f"{'group':<28}{'pairs':>9}{'mean |Δy|':>12}{'median |Δy|':>13}")
    stats = {}
    for name, mask in groups.items():
        cnt = int(mask.sum())
        m = float(dy[mask].mean()) if cnt else float("nan")
        md = float(np.median(dy[mask])) if cnt else float("nan")
        stats[name] = (cnt, m, md)
        print(f"{name.replace(chr(10), ' '):<28}{cnt:>9}{m:>12.4f}{md:>13.4f}")
    print()

    # Per-generator test: pairs whose separation is aligned with one
    # specific generator. Compare measured residuals on both sides.
    g_unit = gens / np.linalg.norm(gens, axis=1, keepdims=True)
    per_gen = []
    for gi, g in enumerate(g_unit):
        proj = D @ g
        cosg = np.abs(proj) / np.maximum(d_tot, 1e-12)
        mask = band & (cosg >= args.align)
        # orient: "upper" mix is the one further along +g
        upper = np.where(proj > 0, ii, jj)
        lower = np.where(proj > 0, jj, ii)
        yu, yl = y[upper[mask]], y[lower[mask]]
        r = np.corrcoef(yl, yu)[0, 1] if mask.sum() > 2 else float("nan")
        per_gen.append((mask, yl, yu, r))
        print(f"Generator {gi+1}: {int(mask.sum())} aligned pairs, "
              f"mean |Δy| = {np.abs(yu-yl).mean():.4f}, corr(y-, y+) = {r:.3f}")

    # Control: pairs aligned with the single most strength-relevant
    # active direction (first right-singular vector of W).
    v1 = row_basis[0]
    proj = D @ v1
    cosv = np.abs(proj) / np.maximum(d_tot, 1e-12)
    ctrl = band & (cosv >= args.align)
    upper = np.where(proj > 0, ii, jj)
    lower = np.where(proj > 0, jj, ii)
    yu_c, yl_c = y[upper[ctrl]], y[lower[ctrl]]
    r_c = np.corrcoef(yl_c, yu_c)[0, 1] if ctrl.sum() > 2 else float("nan")
    print(f"Control (top active dir): {int(ctrl.sum())} aligned pairs, "
          f"mean |Δy| = {np.abs(yu_c-yl_c).mean():.4f}, corr = {r_c:.3f}")

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10))
    fig.suptitle("Do the Discovered Generators Hold in Real Data?\n"
                 "(all axes show MEASURED σ/σ_ideal — no model predictions)",
                 fontweight="bold")

    lims = [min(y.min(), 0.2), y.max() * 1.05]

    def scatter_panel(ax, yl_, yu_, title, color):
        ax.scatter(yl_, yu_, s=14, alpha=0.5, c=color, edgecolors="none")
        ax.plot(lims, lims, "k--", lw=1.5)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("measured σ/σ_ideal  (mix A)")
        ax.set_ylabel("measured σ/σ_ideal  (mix A + ε·g)")
        mae = np.abs(yu_ - yl_).mean() if len(yl_) else float("nan")
        ax.set_title(f"{title}\n{len(yl_)} pairs, mean |Δ| = {mae:.3f}")

    for gi in range(min(3, len(per_gen))):
        mask, yl_, yu_, r = per_gen[gi]
        scatter_panel(axes[0, gi], yl_, yu_,
                      f"Pairs separated along generator g{gi+1}", "#55A868")

    scatter_panel(axes[1, 0], yl_c, yu_c,
                  "CONTROL: pairs separated along the\ntop strength-relevant direction",
                  "#C44E52")

    # Binned |dy| vs distance in each subspace
    ax = axes[1, 1]
    bins = np.linspace(args.dmin, args.dmax, 9)
    mids = 0.5 * (bins[:-1] + bins[1:])
    for mask, dist, label, color in [
            (sym_pairs, d_null, "along symmetry subspace", "#55A868"),
            (act_pairs, d_row, "along active subspace", "#C44E52")]:
        means = [dy[mask & (dist >= lo) & (dist < hi)].mean()
                 if (mask & (dist >= lo) & (dist < hi)).sum() > 5 else np.nan
                 for lo, hi in zip(bins[:-1], bins[1:])]
        ax.plot(mids, means, "o-", color=color, label=label, lw=2)
    if dup_pairs.sum():
        ax.axhline(dy[dup_pairs].mean(), color="gray", ls=":", lw=2,
                   label="noise floor (near-duplicates)")
    ax.set_xlabel("pair separation |Δπ| (standardized)")
    ax.set_ylabel("mean |Δ(σ/σ_ideal)| (measured)")
    ax.set_title("Strength change vs distance moved")
    ax.legend()

    # Summary bars
    ax = axes[1, 2]
    names = list(groups.keys())
    vals = [stats[nm][1] for nm in names]
    cnts = [stats[nm][0] for nm in names]
    colors = ["#8172B3", "#55A868", "#DD8452", "#C44E52"]
    bars = ax.bar(range(len(names)), vals, color=colors, edgecolor="black")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("mean |Δ(σ/σ_ideal)| (measured)")
    ax.set_title("Measured strength-residual change\nby pair type")
    for bar, v, c in zip(bars, vals, cnts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}\n(n={c})", ha="center", va="bottom", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(args.output_dir, "generator_validation.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
