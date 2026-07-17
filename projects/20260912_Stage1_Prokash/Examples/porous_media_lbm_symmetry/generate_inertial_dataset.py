"""
Generate a synthetic porous-media dataset for the INERTIA-DOMINATED region
of the Ergun curve.

The real LBM runs (dataset_lbm_porous.csv) all sit deep in the VISCOUS
(Darcy) branch, Re_p < ~1e-3, where f ~ 150*(1-phi)^2/(Re_p*phi^3).
This script produces the complementary region: Re_p in [1e3, 1e6], where
the modified Reynolds number X = Re_p/(1-phi) >> 85 and the inertial
Forchheimer constant 1.75 dominates (the viscous term contributes < ~5 %).

Each point is drawn from the ACTUAL (textbook) Ergun equation

    f = [ 150*(1-phi)/Re_p + 1.75 ] * (1-phi) / phi^3

with multiplicative log-normal noise on f (default 5 %) and a small
log-normal jitter on the Re_p sampling positions.

Every row carries a physically self-consistent set of the six inputs
(dP_L, v, mu, rho, d, phi) so the Stage1 pipeline can process it exactly
like an LBM row:

    Re_p = rho*v*d/mu       and       f = dP_L*d/(rho*v^2).

The porosities and particle diameters mirror the LBM dataset (6 phi values,
d in {6, 10}); mu and rho are constants matching the LBM lattice fluid.
The CSV schema is identical to dataset_lbm_porous.csv.

Usage:   python generate_inertial_dataset.py [--noise 0.05] [--seed 0]
Output:  dataset_ergun_inertial.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

A_VISC = 150.0     # textbook Ergun viscous constant
B_INER = 1.75      # textbook Ergun inertial constant
_here = os.path.dirname(os.path.abspath(__file__))


def ergun_f(Re, phi):
    """Friction factor from the actual two-term (textbook) Ergun law."""
    return (A_VISC * (1.0 - phi) / Re + B_INER) * (1.0 - phi) / phi**3


def main():
    parser = argparse.ArgumentParser(
        description="Generate inertia-dominated synthetic Ergun dataset")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="Log-normal noise width on f "
                             "(0.05 = ~5%%, 0.2 = ~22%%). Default 0.05")
    parser.add_argument("--re-jitter", type=float, default=0.03,
                        help="Log-normal jitter on each Re_p grid position. "
                             "Default 0.03")
    parser.add_argument("--n-re", type=int, default=30,
                        help="Number of Re_p grid points per (phi, d). "
                             "Default 30")
    parser.add_argument("--re-min-exp", type=float, default=3.0,
                        help="log10 of min Re_p (inertia-dominated only). "
                             "Default 3")
    parser.add_argument("--re-max-exp", type=float, default=6.0,
                        help="log10 of max Re_p. Default 6")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed. Default 0")
    parser.add_argument("--phi-min", type=float, default=None,
                        help="If set, sweep phi in [phi-min, phi-max] with "
                             "n-phi values (log-spaced in porosity) instead "
                             "of mirroring the LBM sweep. Enables the wide-"
                             "porosity variant that exposes the (1-phi)/phi^3 "
                             "curvature.")
    parser.add_argument("--phi-max", type=float, default=None,
                        help="Upper end of the phi sweep (used with --phi-min).")
    parser.add_argument("--n-phi", type=int, default=12,
                        help="Number of phi values when --phi-min/--phi-max "
                             "are given. Default 12.")
    parser.add_argument("--output", default="dataset_ergun_inertial.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Mirror the LBM sweep: same porosities, same particle diameters,
    # same lattice fluid (rho ~ 1, mu = LBM median viscosity).
    lbm = pd.read_csv(os.path.join(_here, "dataset_lbm_porous.csv"))
    if args.phi_min is not None and args.phi_max is not None:
        phis = list(np.linspace(args.phi_min, args.phi_max, args.n_phi))
        print(f"Wide-porosity sweep: phi in [{args.phi_min:.2f}, "
              f"{args.phi_max:.2f}], {args.n_phi} values")
    else:
        phis = sorted(lbm["phi"].round(6).unique())
    ds = sorted(lbm["d"].unique())
    rho0 = 1.0
    mu0 = float(np.median(lbm["mu"]))

    Re_grid = np.logspace(args.re_min_exp, args.re_max_exp, args.n_re)
    print(f"Inertial region: Re_p in [1e{args.re_min_exp:.0f}, "
          f"1e{args.re_max_exp:.0f}]  "
          f"({args.n_re} points x {len(phis)} phi x {len(ds)} d)")
    print(f"Noise width = {args.noise:.3f} "
          f"(~{100 * (np.exp(args.noise) - 1):.0f}% scatter on f), "
          f"Re jitter = {args.re_jitter:.3f}")

    # Sanity: how inertia-dominated is the low end?
    w_visc = (A_VISC * (1 - min(phis)) / Re_grid[0]) / \
             (A_VISC * (1 - min(phis)) / Re_grid[0] + B_INER)
    print(f"Viscous-term share at Re_p = {Re_grid[0]:.0e}: "
          f"{100 * w_visc:.1f}%  (inertial term dominates everywhere)")

    rows = []
    for phv in phis:
        for dv in ds:
            for Re0 in Re_grid:
                Re = float(Re0 * np.exp(rng.normal(0.0, args.re_jitter)))
                v = Re * mu0 / (rho0 * dv)
                f_clean = ergun_f(Re, phv)               # actual Ergun formula
                f_val = float(f_clean * np.exp(rng.normal(0.0, args.noise)))
                dP_L = f_val * rho0 * v**2 / dv          # back out consistent dP_L
                rows.append({
                    "filename": f"ergun_inertial_phi{phv:.3f}_d{int(dv)}_Re{Re0:.1e}",
                    "tau": np.nan, "delta_p": np.nan,
                    "dP_L": dP_L, "v": v, "mu": mu0, "rho": rho0,
                    "d": dv, "phi": phv, "f": f_val, "Re_p": Re,
                    "f_ergun": f_clean,
                    "steps": 0, "converged": "TRUE", "stalled": "FALSE",
                })
    syn = pd.DataFrame(rows)

    out = args.output if os.path.isabs(args.output) \
        else os.path.join(_here, args.output)
    syn.to_csv(out, index=False)

    print(f"\nRows: {len(syn)}  "
          f"(Re_p {syn['Re_p'].min():.2e} .. {syn['Re_p'].max():.2e},  "
          f"f {syn['f'].min():.2f} .. {syn['f'].max():.2f})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
