"""Compare LBM data against the full two-term textbook Ergun equation.

Left panel:  Ergun collapse coordinates over an extended Re range, with the
             viscous term (150/x), inertial term (1.75) and full curve drawn
             separately, so it is visible which term the LBM data actually
             probes.
Right panel: compensated view A_eff = f*phi^3*Re_p/(1-phi)^2.  In these
             coordinates the textbook curve is 150 + 1.75*x, i.e. the viscous
             constant appears as a level and the inertial term as the rising
             tail.  The LBM points read off directly as an effective viscous
             constant.

Usage:  python plot_ergun_two_terms.py [--data dataset_lbm_porous.csv]
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":             15,
    "axes.titlesize":        18,
    "axes.labelsize":        17,
    "xtick.labelsize":       15,
    "ytick.labelsize":       15,
    "legend.fontsize":       13,
})

A_VISC = 150.0   # textbook Ergun viscous constant
B_INER = 1.75    # textbook Ergun inertial constant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset_lbm_porous.csv")
    parser.add_argument("--output", default="lbm_ergun_two_terms.png")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    phi = df["phi"].to_numpy()
    x = df["Re_p"].to_numpy() / (1.0 - phi)          # modified Reynolds number
    y = df["f"].to_numpy() * phi**3 / (1.0 - phi)    # modified friction factor
    a_eff = y * x                                     # compensated: A + B*x form

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Textbook Ergun (both terms) vs LBM", fontweight="bold", fontsize=21)

    phi_rounded = np.round(phi, 3)
    uniq = np.unique(phi_rounded)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(uniq)))

    # ── Panel 1: full Ergun curve, term by term, extended Re range ──────────
    x_line = np.logspace(-7, 4, 400)
    ax1.loglog(x_line, A_VISC / x_line + B_INER, "k-", lw=2.5,
               label=f"Full Ergun: {A_VISC:.0f}/x + {B_INER}")
    ax1.loglog(x_line, A_VISC / x_line, "--", color="#4C72B0", lw=2,
               label=f"Viscous term: {A_VISC:.0f}/x")
    ax1.axhline(B_INER, ls=":", color="#DD8452", lw=2,
                label=f"Inertial term: {B_INER}")
    x_cross = A_VISC / B_INER
    ax1.axvline(x_cross, color="gray", lw=1, alpha=0.6)
    ax1.text(x_cross * 1.5, 3e7, f"terms equal\nat x ≈ {x_cross:.0f}",
             fontsize=12, color="gray")

    for c, phi_val in zip(colors, uniq):
        mask = phi_rounded == phi_val
        ax1.loglog(x[mask], y[mask], "o", ms=5, color=c, mec="black",
                   mew=0.3, alpha=0.8, label=f"LBM φ ≈ {phi_val:.3f}")

    ax1.set_xlabel(r"$Re_p\,/\,(1-\phi)$")
    ax1.set_ylabel(r"$f\cdot\phi^3\,/\,(1-\phi)$")
    ax1.set_title("LBM data probes only the viscous branch")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(loc="lower left", framealpha=0.9)

    # ── Panel 2: compensated plot — effective viscous constant ─────────────
    x_line2 = np.logspace(np.log10(x.min() / 2), np.log10(x.max() * 2), 200)
    ax2.semilogx(x_line2, A_VISC + B_INER * x_line2, "k-", lw=2.5,
                 label=f"Textbook: {A_VISC:.0f} + {B_INER}·x")

    a_mean = a_eff.mean()
    ax2.axhline(a_mean, ls="--", color="#C44E52", lw=2,
                label=f"LBM mean: A ≈ {a_mean:.0f}")

    for c, phi_val in zip(colors, uniq):
        mask = phi_rounded == phi_val
        ax2.semilogx(x[mask], a_eff[mask], "o", ms=6, color=c, mec="black",
                     mew=0.3, alpha=0.8)

    ax2.set_xlabel(r"$Re_p\,/\,(1-\phi)$")
    ax2.set_ylabel(r"$A_{eff} = f\cdot\phi^3\,Re_p\,/\,(1-\phi)^2$")
    ax2.set_title("Compensated: effective Ergun constant")
    ax2.set_ylim(0, 200)
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.text(0.03, 0.06,
             f"Inertial term adds ≤ {B_INER * x.max():.3f} here\n"
             f"(≤ {100 * B_INER * x.max() / A_VISC:.3f}% of textbook level)",
             transform=ax2.transAxes, fontsize=12, color="dimgray")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {args.output}")
    print(f"LBM effective viscous constant: {a_mean:.1f} (textbook {A_VISC:.0f}, "
          f"ratio {a_mean / A_VISC:.3f})")
    print(f"Max inertial contribution in data range: {B_INER * x.max():.4f}")


if __name__ == "__main__":
    main()
