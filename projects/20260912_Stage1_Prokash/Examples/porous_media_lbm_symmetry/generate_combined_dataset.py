"""
Build a combined porous-media dataset that spans the FULL Ergun curve:

  * the real LBM runs (deep Darcy / viscous branch, Re_p < 1e-3), plus
  * synthetic points over the REST of the range (Re_p ~ 1e-3 .. 1e6),
    generated from the TEXTBOOK Ergun equation with multiplicative noise.

The synthetic points use the textbook constants (150, 1.75):

    Y = f * phi^3 / (1-phi) = 150 / X + 1.75,   X = Re_p / (1-phi).

Note: the real LBM data sits at an EFFECTIVE viscous constant
A_eff ~ 0.65 * 150 (LBM gives ~35% lower drag), so on this plot the LBM
points lie slightly BELOW the textbook viscous branch — they are the
real measured data and are marked distinctly.

Every synthetic row carries a physically self-consistent set of the six
inputs (dP_L, v, mu, rho, d, phi) so the Stage1 dimensional-analysis
pipeline can process it exactly like an LBM row:

    Re_p = rho*v*d/mu      and      f = dP_L*d/(rho*v^2).

Usage:  python generate_combined_dataset.py
Output: dataset_combined_ergun.csv
"""

import os
import numpy as np
import pandas as pd

A_VISC = 150.0     # textbook Ergun viscous constant
B_INER = 1.75      # textbook Ergun inertial constant
_here = os.path.dirname(os.path.abspath(__file__))


def ergun_f(Re, phi, a_visc, b_iner):
    """Friction factor from the (effective) two-term Ergun law."""
    X = Re / (1.0 - phi)
    Y = a_visc / X + b_iner
    return Y * (1.0 - phi) / phi**3


def main():
    rng = np.random.default_rng(0)
    src = os.path.join(_here, "dataset_lbm_porous.csv")
    df = pd.read_csv(src)

    ok = (df["converged"].astype(str).str.lower() == "true") & \
         (df["stalled"].astype(str).str.lower() != "true")
    lbm = df[ok].copy()
    lbm["source"] = "lbm"

    # --- fit the LBM effective constants (viscous branch) ---
    phi = lbm["phi"].to_numpy()
    X = lbm["Re_p"].to_numpy() / (1.0 - phi)
    Y = lbm["f"].to_numpy() * phi**3 / (1.0 - phi)
    A_eff = float(np.median(Y * X))          # effective viscous constant (~99)
    ratio = A_eff / A_VISC                    # ~0.66
    print(f"LBM effective viscous constant A_eff = {A_eff:.1f} "
          f"(textbook {A_VISC:.0f}, ratio {ratio:.3f})")
    print(f"Synthetic data uses TEXTBOOK Ergun constants ({A_VISC:.0f}, {B_INER})")

    # --- synthetic rows spanning the REST of the curve ---
    # LBM already covers Re_p ~ 1e-7 .. 1e-3 (viscous). We fill everything
    # above that with a WIDE range: the transition knee (Re_p ~ 85) and the
    # full inertial plateau, so the combined data covers the whole Ergun
    # curve. Synthetic f uses the TEXTBOOK Ergun equation (150, 1.75).
    phis = sorted(lbm["phi"].round(6).unique())        # 6 porosities
    ds = sorted(lbm["d"].unique())                     # {6, 10}
    rho0 = 1.0
    mu0 = float(np.median(lbm["mu"]))                  # ~0.133 (LBM viscosity)
    Re_grid = np.logspace(-3.0, 6.0, 30)               # 1e-3 .. 1e6 (wide: transition+inertial)
    noise_sigma = 0.05                                 # 5% multiplicative noise

    rows = []
    for phv in phis:
        for dv in ds:
            for Re0 in Re_grid:
                Re = float(Re0 * np.exp(rng.normal(0.0, 0.03)))   # jitter Re
                v = Re * mu0 / (rho0 * dv)
                f_clean = ergun_f(Re, phv, A_VISC, B_INER)        # TEXTBOOK Ergun
                f_val = float(f_clean * np.exp(rng.normal(0.0, noise_sigma)))
                dP_L = f_val * rho0 * v**2 / dv
                f_ergun_txt = (A_VISC * (1.0 - phv) / Re + B_INER) * (1.0 - phv) / phv**3
                rows.append({
                    "filename": f"synthetic_ergun_phi{phv:.3f}_d{int(dv)}_Re{Re0:.1e}",
                    "tau": np.nan, "delta_p": np.nan,
                    "dP_L": dP_L, "v": v, "mu": mu0, "rho": rho0,
                    "d": dv, "phi": phv, "f": f_val, "Re_p": Re,
                    "f_ergun": f_ergun_txt,
                    "steps": 0, "converged": "TRUE", "stalled": "FALSE",
                    "source": "synthetic",
                })
    syn = pd.DataFrame(rows)

    combined = pd.concat([lbm, syn], ignore_index=True)
    out = os.path.join(_here, "dataset_combined_ergun.csv")
    combined.to_csv(out, index=False)

    print(f"\nLBM rows:       {len(lbm)}  "
          f"(Re_p {lbm['Re_p'].min():.1e} .. {lbm['Re_p'].max():.1e})")
    print(f"Synthetic rows: {len(syn)}  "
          f"(Re_p {syn['Re_p'].min():.1e} .. {syn['Re_p'].max():.1e})")
    print(f"Combined rows:  {len(combined)}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
