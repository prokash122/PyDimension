"""
Publication figure for the discovered strength-preserving generators.

One claim, three panels, measured data only:

  (a) WHAT the generators are: signed components of g1..g3 in the
      dimensionless mix-ratio space.
  (b) EVIDENCE: for pairs of real mixes whose composition difference
      points along a generator, the measured strength ratio of mix B
      vs mix A lies on the 1:1 line; pairs separated along the model's
      strength-relevant direction fall off it.
  (c) SUMMARY: mean |difference| in measured strength ratio per pair
      type, with bootstrap 95% CIs, against the replicate noise floor.

No model prediction appears anywhere: both axes of (b) and all bars of
(c) are measured strengths (normalized by the Yeh 1998 baseline).

Usage
-----
    python make_publication_figure.py   # after discover_symmetry_dimensionless.py
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

# Validated palette (dataviz reference instance, light mode)
BLUE = "#2a78d6"      # generator pairs / generator bars
RED = "#e34948"       # strength-direction control
GRAY = "#898781"      # muted / random pairs
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "text.color":      INK,
    "axes.labelcolor": INK2,
    "xtick.color":     INK2,
    "ytick.color":     INK2,
    "axes.edgecolor":  BASE,
    "axes.linewidth":  0.8,
})

_here = os.path.dirname(os.path.abspath(__file__))

PI_LABELS = ["w/b", "FA/b", "Cem/b", "SP/b", "CA/b", "FiA/b", "ln(t/28)"]


def bootstrap_ci(vals, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return np.percentile(means, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts",
                        default="output_concrete_dimensionless/pipeline_artifacts.npz")
    parser.add_argument("--output-dir", default="output_concrete_dimensionless")
    parser.add_argument("--align", type=float, default=0.90)
    parser.add_argument("--dmin", type=float, default=0.5)
    parser.add_argument("--dmax", type=float, default=2.5)
    parser.add_argument("--dup", type=float, default=0.05)
    args = parser.parse_args()

    path = args.artifacts
    if not os.path.exists(path):
        path = os.path.join(_here, args.artifacts)
    data = np.load(path, allow_pickle=True)
    X = data["X_norm"]
    y = data["y"]
    W = data["W"]
    gens = data["generators"]

    n = X.shape[0]
    g_unit = gens / np.linalg.norm(gens, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(W)
    v1 = Vt[0]

    ii, jj = np.triu_indices(n, 1)
    D = X[ii] - X[jj]
    d_tot = np.linalg.norm(D, axis=1)
    band = (d_tot >= args.dmin) & (d_tot <= args.dmax)

    def aligned_pairs(direction):
        proj = D @ direction
        cos = np.abs(proj) / np.maximum(d_tot, 1e-12)
        mask = band & (cos >= args.align)
        lower = np.where(proj > 0, jj, ii)
        upper = np.where(proj > 0, ii, jj)
        return y[lower[mask]], y[upper[mask]]

    # Pool the three generators (union of aligned pairs).
    gen_masks = []
    for g in g_unit:
        proj = D @ g
        cos = np.abs(proj) / np.maximum(d_tot, 1e-12)
        gen_masks.append(band & (cos >= args.align))
    gen_mask = np.logical_or.reduce(gen_masks)
    # Orientation for the scatter: along the generator with best alignment
    yl_g, yu_g = y[ii[gen_mask]], y[jj[gen_mask]]
    dy_gen = np.abs(yu_g - yl_g)

    yl_c, yu_c = aligned_pairs(v1)
    dy_ctrl = np.abs(yu_c - yl_c)

    dy_rand = np.abs(y[ii[band]] - y[jj[band]])
    dup_mask = d_tot <= args.dup
    dy_dup = np.abs(y[ii[dup_mask]] - y[jj[dup_mask]])

    print(f"generator pairs: n={gen_mask.sum()}, mean |dy|={dy_gen.mean():.3f}")
    print(f"control pairs:   n={len(dy_ctrl)}, mean |dy|={dy_ctrl.mean():.3f}")
    print(f"random pairs:    n={band.sum()}, mean |dy|={dy_rand.mean():.3f}")
    print(f"replicates:      n={dup_mask.sum()}, mean |dy|={dy_dup.mean():.3f}")

    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3),
                             gridspec_kw={"width_ratios": [1.05, 1.0, 0.85]})
    fig.patch.set_facecolor("white")

    # ---------------- (a) the generators ----------------
    ax = axes[0]
    ypos = np.arange(len(PI_LABELS))[::-1]
    height = 0.26
    shades = [BLUE, "#6da7ec", "#104281"]     # blue ramp steps 300/650
    for gi, (g, sh) in enumerate(zip(g_unit, shades)):
        ax.barh(ypos + (1 - gi) * height, g, height=height * 0.92,
                color=sh, edgecolor="white", linewidth=0.5,
                label=f"g{gi+1}")
    ax.axvline(0, color=BASE, lw=0.8)
    ax.set_yticks(ypos + 0)
    ax.set_yticklabels(PI_LABELS)
    ax.set_xlabel("generator component (standardized units)")
    ax.set_title("(a)  Discovered strength-preserving\ndirections (generators)",
                 loc="left")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---------------- (b) evidence scatter ----------------
    ax = axes[1]
    lims = [0.25, 2.0]
    ax.plot(lims, lims, ls="--", lw=1.2, color=INK2, zorder=1)
    ax.scatter(yl_c, yu_c, s=22, alpha=0.55, color=RED, edgecolors="none",
               zorder=2, label=f"pairs along strength direction (n={len(yl_c)})")
    ax.scatter(yl_g, yu_g, s=22, alpha=0.55, color=BLUE, edgecolors="none",
               zorder=3, label=f"pairs along a generator (n={len(yl_g)})")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("measured  σc/σideal   —  mix A")
    ax.set_ylabel("measured  σc/σideal   —  mix B")
    ax.set_title("(b)  Real mix pairs: measured\nstrength of B vs A", loc="left")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.1)
    ax.annotate("1:1", xy=(1.75, 1.79), color=INK2, fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # ---------------- (c) summary bars ----------------
    ax = axes[2]
    groups = [
        ("replicate\nmixes", dy_dup, GRAY),
        ("along a\ngenerator", dy_gen, BLUE),
        ("random\npairs", dy_rand, GRAY),
        ("along strength\ndirection", dy_ctrl, RED),
    ]
    xs = np.arange(len(groups))
    for x, (label, vals, color) in zip(xs, groups):
        m = vals.mean()
        lo, hi = bootstrap_ci(vals)
        ax.bar(x, m, width=0.62, color=color, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK,
                    lw=1.2, capsize=3)
        ax.text(x, hi + 0.012, f"{m:.2f}", ha="center", color=INK2, fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=9.5)
    ax.set_ylabel("mean |Δ(σc/σideal)| between pair,  measured")
    ax.set_title("(c)  Measured strength change\nby pair type (95% CI)",
                 loc="left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    plt.tight_layout(w_pad=2.0)
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "publication_figure.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
