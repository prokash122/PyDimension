"""
Overview plot of the two regimes used in this example, in Ergun collapse
coordinates  Y = f*phi^3/(1-phi)  vs  X = Re_p/(1-phi):

  * VISCOUS region — the real LBM runs (dataset_lbm_porous.csv),
    X < ~2e-3, where Y ~ 150/X (Darcy branch).
  * INERTIAL region — synthetic points from the actual Ergun formula
    with noise (dataset_ergun_inertial.csv), X > ~2e3, where Y ~ 1.75.

The textbook master curve 150/X + 1.75 is drawn across the whole range.
The full viscous side (up to the crossover X* = 150/1.75 ~ 86, where the
two Ergun terms are equal) is shaded blue; the inertial side beyond X*
is shaded orange. The LBM points sit ~0.65x below the textbook viscous
branch (the LBM gives ~35 % lower drag than the empirical constant 150);
the synthetic inertial points scatter around the textbook plateau by
construction.

Usage:   python plot_two_regions.py
Output:  ergun_two_regions.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.size":       24,
    "axes.titlesize":  30,
    "axes.labelsize":  29,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 23,
})

X_CROSS = 150.0 / 1.75          # viscous term == inertial term


def load(name):
    df = pd.read_csv(os.path.join(_here, name))
    ok = (df["converged"].astype(str).str.lower() == "true") & \
         (df["stalled"].astype(str).str.lower() != "true")
    df = df[ok]
    X = df["Re_p"] / (1.0 - df["phi"])
    Y = df["f"] * df["phi"]**3 / (1.0 - df["phi"])
    return X.to_numpy(), Y.to_numpy(), df["phi"].to_numpy()


def main():
    Xv, Yv, phiv = load("dataset_lbm_porous.csv")
    Xi, Yi, phii = load("dataset_ergun_inertial.csv")

    fig, ax = plt.subplots(figsize=(15, 9.5))

    x_lo = 10 ** (np.log10(Xv.min()) - 0.4)
    x_hi = 10 ** (np.log10(Xi.max()) + 0.4)
    x_curve = np.logspace(np.log10(x_lo), np.log10(x_hi), 400)
    ax.loglog(x_curve, 150.0 / x_curve + 1.75, "k--", lw=2.6,
              label="Actual Ergun: $150/X + 1.75$")

    ax.loglog(Xv, Yv, "o", ms=8, alpha=0.8, color="#4C72B0",
              markeredgecolor="black", markeredgewidth=0.4,
              label=f"LBM data — viscous region (n={len(Xv)})")
    ax.loglog(Xi, Yi, "^", ms=8, alpha=0.8, color="#DD8452",
              markeredgecolor="black", markeredgewidth=0.4,
              label=f"Synthetic Ergun + noise — inertial region (n={len(Xi)})")

    # Shade the FULL viscous side (up to the term crossover X*) in blue,
    # and the inertial side beyond X* in orange.
    ax.axvspan(x_lo, X_CROSS, color="#4C72B0", alpha=0.10)
    ax.axvspan(X_CROSS, x_hi, color="#DD8452", alpha=0.08)
    ax.axvline(X_CROSS, color="grey", ls=":", lw=2.0)

    ax.text(X_CROSS, 10 ** (np.log10(Yi.min()) + 0.35),
            "  $X^* = 150/1.75 \\approx 86$",
            ha="left", va="bottom", color="#555555", fontsize=22)

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel(r"$X = Re_p / (1-\phi)$")
    ax.set_ylabel(r"$Y = f \cdot \phi^3 / (1-\phi)$")
    ax.set_title("Ergun master curve — the two regions studied separately",
                 fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    out = os.path.join(_here, "ergun_two_regions.png")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
