"""
Generate a synthetic porous-media dataset for the VISCOUS-DOMINATED
(Darcy) region of the Ergun curve, mirroring generate_inertial_dataset.py
but on the low-Re side.

The real LBM runs (dataset_lbm_porous.csv) sit in this region already, but
their porosity range is narrow (phi in [0.46, 0.61]), which triggers the
phi <-> (1-phi) identifiability wall.  This script fills the same
Re_p < 1e-3 region with the actual textbook Ergun formula

    f = [ 150*(1-phi)/Re_p + 1.75 ] * (1-phi) / phi^3

evaluated over a WIDE porosity range with multiplicative log-normal noise
on f and a small log-normal jitter on the Re_p sampling positions.

Every row carries a physically self-consistent set of the six inputs
(dP_L, v, mu, rho, d, phi) so the Stage1 pipeline can process it exactly
like an LBM row.

Usage:  python generate_viscous_dataset.py --phi-min 0.15 --phi-max 0.85
Output: dataset_ergun_viscous.csv  (or dataset_ergun_viscous_widephi.csv)
"""

import os
import argparse
import numpy as np
import pandas as pd

A_VISC = 150.0
B_INER = 1.75
_here = os.path.dirname(os.path.abspath(__file__))


def ergun_f(Re, phi):
    return (A_VISC * (1.0 - phi) / Re + B_INER) * (1.0 - phi) / phi**3


def main():
    parser = argparse.ArgumentParser(
        description="Generate viscous-dominated synthetic Ergun dataset")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="Log-normal noise width on f. Default 0.05")
    parser.add_argument("--re-jitter", type=float, default=0.03,
                        help="Log-normal jitter on each Re_p position. "
                             "Default 0.03")
    parser.add_argument("--n-re", type=int, default=30,
                        help="Number of Re_p grid points per (phi, d). "
                             "Default 30")
    parser.add_argument("--re-min-exp", type=float, default=-6.0,
                        help="log10 of min Re_p (deep-viscous). Default -6")
    parser.add_argument("--re-max-exp", type=float, default=-3.0,
                        help="log10 of max Re_p (still viscous-dominated). "
                             "Default -3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi-min", type=float, default=None,
                        help="If set, sweep phi in [phi-min, phi-max] with "
                             "n-phi values.  Enables the wide-porosity "
                             "variant.")
    parser.add_argument("--phi-max", type=float, default=None)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument("--output", default="dataset_ergun_viscous.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

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
    print(f"Viscous region: Re_p in [1e{args.re_min_exp:.0f}, "
          f"1e{args.re_max_exp:.0f}]  "
          f"({args.n_re} x {len(phis)} phi x {len(ds)} d)")
    print(f"Noise width = {args.noise:.3f}, Re jitter = {args.re_jitter:.3f}")

    # Sanity: at the high end of Re_p is the viscous term still dominating?
    w_visc = (A_VISC * (1 - min(phis)) / Re_grid[-1]) / \
             (A_VISC * (1 - min(phis)) / Re_grid[-1] + B_INER)
    print(f"Viscous-term share at Re_p = {Re_grid[-1]:.0e}: "
          f"{100 * w_visc:.1f}%  (should be >>50% throughout)")

    rows = []
    for phv in phis:
        for dv in ds:
            for Re0 in Re_grid:
                Re = float(Re0 * np.exp(rng.normal(0.0, args.re_jitter)))
                v = Re * mu0 / (rho0 * dv)
                f_clean = ergun_f(Re, phv)
                f_val = float(f_clean * np.exp(rng.normal(0.0, args.noise)))
                dP_L = f_val * rho0 * v**2 / dv
                rows.append({
                    "filename": f"ergun_viscous_phi{phv:.3f}_d{int(dv)}_Re{Re0:.1e}",
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
          f"f {syn['f'].min():.2e} .. {syn['f'].max():.2e})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
