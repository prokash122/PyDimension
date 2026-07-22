"""
Generate a synthetic porous-media dataset spanning BOTH Ergun regimes in a
single CSV — the input for the regime-aware (clustering dimensionless
learning) pipeline in discover_regimes_clustering.py.

Unlike generate_viscous_dataset.py / generate_inertial_dataset.py, which
sample only one limiting branch each, this script sweeps Re_p continuously
across the full curve

    f = [ 150*(1-phi)/Re_p + 1.75 ] * (1-phi) / phi^3

so the viscous (Darcy) regime, the transition zone, and the inertial
(Forchheimer) regime all coexist in one dataset.  No regime label is fed
to the discovery pipeline; a reference column `regime_true` (from the
analytic crossover Re_p/(1-phi) = 150/1.75 ~ 85.7) is saved for
validation only.

Every row carries a physically self-consistent set of the six inputs
(dP_L, v, mu, rho, d, phi) so the Stage1 pipeline can process it exactly
like an LBM row.

Usage:  python generate_combined_dataset.py --phi-min 0.15 --phi-max 0.85
Output: dataset_ergun_combined_widephi.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

A_VISC = 150.0
B_INER = 1.75
RE_CROSS = A_VISC / B_INER          # Re_p/(1-phi) where both terms are equal
_here = os.path.dirname(os.path.abspath(__file__))


def ergun_f(Re, phi):
    return (A_VISC * (1.0 - phi) / Re + B_INER) * (1.0 - phi) / phi**3


def main():
    parser = argparse.ArgumentParser(
        description="Generate combined two-regime synthetic Ergun dataset")
    parser.add_argument("--noise", type=float, default=0.05,
                        help="Log-normal noise width on f. Default 0.05")
    parser.add_argument("--re-jitter", type=float, default=0.03,
                        help="Log-normal jitter on each Re_p position. "
                             "Default 0.03")
    parser.add_argument("--n-re", type=int, default=60,
                        help="Number of Re_p grid points per (phi, d). "
                             "Default 60")
    parser.add_argument("--re-min-exp", type=float, default=-6.0,
                        help="log10 of min Re_p (deep-viscous). Default -6")
    parser.add_argument("--re-max-exp", type=float, default=6.0,
                        help="log10 of max Re_p (deep-inertial). Default 6")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--phi-min", type=float, default=0.15)
    parser.add_argument("--phi-max", type=float, default=0.85)
    parser.add_argument("--n-phi", type=int, default=12)
    parser.add_argument("--output",
                        default="dataset_ergun_combined_widephi.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    lbm = pd.read_csv(os.path.join(_here, "dataset_lbm_porous.csv"))
    phis = list(np.linspace(args.phi_min, args.phi_max, args.n_phi))
    ds = sorted(lbm["d"].unique())
    rho0 = 1.0
    mu0 = float(np.median(lbm["mu"]))

    Re_grid = np.logspace(args.re_min_exp, args.re_max_exp, args.n_re)
    print(f"Combined sweep: Re_p in [1e{args.re_min_exp:.0f}, "
          f"1e{args.re_max_exp:.0f}]  "
          f"({args.n_re} x {len(phis)} phi x {len(ds)} d)")
    print(f"phi in [{args.phi_min:.2f}, {args.phi_max:.2f}], "
          f"{args.n_phi} values")
    print(f"Noise width = {args.noise:.3f}, Re jitter = {args.re_jitter:.3f}")
    print(f"Analytic crossover: Re_p/(1-phi) = {RE_CROSS:.1f}")

    rows = []
    for phv in phis:
        for dv in ds:
            for Re0 in Re_grid:
                Re = float(Re0 * np.exp(rng.normal(0.0, args.re_jitter)))
                v = Re * mu0 / (rho0 * dv)
                f_clean = ergun_f(Re, phv)
                f_val = float(f_clean * np.exp(rng.normal(0.0, args.noise)))
                dP_L = f_val * rho0 * v**2 / dv
                regime = "viscous" if Re / (1.0 - phv) < RE_CROSS \
                    else "inertial"
                rows.append({
                    "filename": f"ergun_comb_phi{phv:.3f}_d{int(dv)}_Re{Re0:.1e}",
                    "tau": np.nan, "delta_p": np.nan,
                    "dP_L": dP_L, "v": v, "mu": mu0, "rho": rho0,
                    "d": dv, "phi": phv, "f": f_val, "Re_p": Re,
                    "f_ergun": f_clean, "regime_true": regime,
                    "steps": 0, "converged": "TRUE", "stalled": "FALSE",
                })
    syn = pd.DataFrame(rows)

    out = args.output if os.path.isabs(args.output) \
        else os.path.join(_here, args.output)
    syn.to_csv(out, index=False)

    n_visc = int((syn["regime_true"] == "viscous").sum())
    print(f"\nRows: {len(syn)}  "
          f"(viscous {n_visc}, inertial {len(syn) - n_visc})")
    print(f"Re_p {syn['Re_p'].min():.2e} .. {syn['Re_p'].max():.2e},  "
          f"f {syn['f'].min():.2e} .. {syn['f'].max():.2e}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
