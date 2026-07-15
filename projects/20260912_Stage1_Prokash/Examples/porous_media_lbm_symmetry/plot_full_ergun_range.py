"""
Plot the FULL Ergun curve in collapse coordinates with both datasets on
the single master curve:

    X = Re_p / (1-phi),   Y = f * phi^3 / (1-phi),   Y = A_eff/X + B_eff

  * LBM runs        -> viscous branch (Y ~ A_eff/X), low X
  * synthetic Ergun -> inertial branch + plateau (Y -> B_eff), high X

Usage:  python plot_full_ergun_range.py [--data dataset_combined_ergun.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated palette (dataviz reference instance, light mode)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       15,
    "axes.titlesize":  18,
    "axes.labelsize":  17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "text.color":      INK,
    "axes.labelcolor": INK2,
    "xtick.color":     INK2,
    "ytick.color":     INK2,
    "axes.edgecolor":  BASE,
})

A_VISC = 150.0
B_INER = 1.75
_here = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_combined_ergun.csv")
    parser.add_argument("--output", default="combined_ergun_full_range.png")
    args = parser.parse_args()

    path = args.data if os.path.exists(args.data) else os.path.join(_here, args.data)
    df = pd.read_csv(path)
    src = df.get("source", pd.Series(["lbm"] * len(df)))

    phi = df["phi"].to_numpy()
    X = df["Re_p"].to_numpy() / (1.0 - phi)
    Y = df["f"].to_numpy() * phi**3 / (1.0 - phi)

    lbm = (src == "lbm").to_numpy()
    syn = (src == "synthetic_inertial").to_numpy()

    # effective constants fit from the LBM viscous branch
    A_eff = float(np.median((Y * X)[lbm]))
    ratio = A_eff / A_VISC
    B_eff = B_INER * ratio

    fig, ax = plt.subplots(figsize=(11, 7.5))
    fig.suptitle("Full Ergun curve — LBM (viscous) + synthetic (inertial) on one curve",
                 fontweight="bold", fontsize=18)

    x_line = np.logspace(np.log10(X.min()) - 0.5, np.log10(X.max()) + 0.5, 500)
    ax.loglog(x_line, A_eff / x_line + B_eff, "-", color=INK, lw=2.5, zorder=3,
              label=f"effective Ergun:  {A_eff:.0f}/X + {B_eff:.2f}")
    ax.loglog(x_line, A_eff / x_line, "--", color=INK2, lw=1.6, alpha=0.8, zorder=2,
              label=f"viscous term:  {A_eff:.0f}/X")
    ax.axhline(B_eff, ls=":", color=INK2, lw=1.6, alpha=0.8, zorder=2,
               label=f"inertial plateau:  {B_eff:.2f}")
    x_cross = A_eff / B_eff
    ax.axvline(x_cross, color=BASE, lw=1.2, zorder=1)
    ax.text(x_cross * 1.3, Y.max() * 0.3, f"terms equal\nat X ≈ {x_cross:.0f}",
            fontsize=12, color=INK2)

    ax.scatter(X[lbm], Y[lbm], s=34, color=BLUE, edgecolors="white", linewidths=0.4,
               alpha=0.9, zorder=5, label=f"LBM data (viscous, n={lbm.sum()})")
    ax.scatter(X[syn], Y[syn], s=34, color=ORANGE, edgecolors="white", linewidths=0.4,
               alpha=0.9, zorder=5, label=f"synthetic Ergun (inertial, n={syn.sum()})")

    ax.set_xlabel(r"$X = Re_p\,/\,(1-\phi)$")
    ax.set_ylabel(r"$Y = f\cdot\phi^3\,/\,(1-\phi)$")
    ax.grid(True, which="both", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = args.output if os.path.isabs(args.output) else os.path.join(_here, args.output)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")
    print(f"A_eff={A_eff:.1f}, B_eff={B_eff:.3f}, X range {X.min():.1e}..{X.max():.1e}")


if __name__ == "__main__":
    main()
