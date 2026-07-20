"""
Generate a synthetic pipe-friction dataset spanning the ENTIRE Moody
chart — laminar branch, smooth-turbulent (Blasius) zone, and the fully
rough limit — from the textbook ground truth:

    laminar   (Re < 2300)  :  f = 64 / Re                      [exact]
    turbulent (Re > 4000)  :  Colebrook-White
        1/sqrt(f) = -2 log10( (eps/D)/3.7 + 2.51/(Re sqrt(f)) )

The critical zone 2300 < Re < 4000 is excluded from sampling, exactly as
it is left blank on a Moody chart: the flow there is intermittent and no
steady correlation applies.

This is the input for the 2-D region-free local-symmetry scan
(discover_local_symmetry_2d.py).  NOTHING in the pipeline-facing columns
tells the scan where the regimes are; the known local scaling laws it
should rediscover are

    laminar        : f = 64 * Re^-1            (eps/D exponent 0)
    smooth turb.   : f ~ 0.316 * Re^-1/4       (Blasius; eps/D exponent 0)
    fully rough    : f ~ Re^0, d ln f/d ln(eps/D) = 2/ln(3.7 D/eps)
                     (von Karman log-law; ~0.19...0.34 over this sweep)

Every row carries a physically self-consistent set of the raw inputs
(dP_L, v, mu, rho, D, eps) for the Darcy-Weisbach friction factor
f = 2 * dP_L * D / (rho v^2), so the Stage1 pipeline can process it like
any other dataset.  `branch` and `f_true` are reference-only columns.

Usage:  python generate_moody_dataset.py
Output: dataset_moody.csv
"""

import os
import argparse
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))

RE_LAM_MAX = 2300.0
RE_TURB_MIN = 4000.0


def colebrook_f(Re, eps_rel, n_iter=50):
    """Vectorised fixed-point solve of Colebrook-White."""
    Re = np.asarray(Re, dtype=float)
    eps_rel = np.asarray(eps_rel, dtype=float)
    inv_sqrt_f = np.full_like(Re, 4.0)  # 1/sqrt(f), f ~ 0.06 start
    for _ in range(n_iter):
        inv_sqrt_f = -2.0 * np.log10(eps_rel / 3.7
                                     + 2.51 * inv_sqrt_f / Re)
    return 1.0 / inv_sqrt_f**2


def moody_f(Re, eps_rel):
    lam = Re < RE_LAM_MAX
    f = np.where(lam, 64.0 / Re, colebrook_f(Re, eps_rel))
    return f, np.where(lam, "laminar", "colebrook")


def main():
    parser = argparse.ArgumentParser(
        description="Generate full Moody-chart synthetic dataset")
    parser.add_argument("--noise", type=float, default=0.03,
                        help="Log-normal noise width on f. Default 0.03")
    parser.add_argument("--re-jitter", type=float, default=0.03,
                        help="Log-normal jitter on each Re position.")
    parser.add_argument("--n-re", type=int, default=100,
                        help="Re grid points before removing the critical "
                             "zone. Default 100")
    parser.add_argument("--n-eps", type=int, default=40,
                        help="Relative-roughness grid points. Default 40")
    parser.add_argument("--re-min", type=float, default=3e2)
    parser.add_argument("--re-max", type=float, default=1e8)
    parser.add_argument("--eps-min", type=float, default=1e-6)
    parser.add_argument("--eps-max", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="dataset_moody.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    Re_grid = np.logspace(np.log10(args.re_min), np.log10(args.re_max),
                          args.n_re)
    keep = (Re_grid < RE_LAM_MAX) | (Re_grid > RE_TURB_MIN)
    Re_grid = Re_grid[keep]
    eps_grid = np.logspace(np.log10(args.eps_min), np.log10(args.eps_max),
                           args.n_eps)

    print(f"Moody sweep: Re in [{args.re_min:.0e}, {args.re_max:.0e}] "
          f"({len(Re_grid)} values, critical zone "
          f"{RE_LAM_MAX:.0f}..{RE_TURB_MIN:.0f} excluded)")
    print(f"eps/D in [{args.eps_min:.0e}, {args.eps_max:.0e}] "
          f"({args.n_eps} values)")
    print(f"Noise width = {args.noise:.3f}, Re jitter = {args.re_jitter:.3f}")

    rho0 = 1000.0   # water
    mu0 = 1.0e-3
    D0 = 0.1        # pipe diameter [m]

    rows = []
    for er in eps_grid:
        for Re0 in Re_grid:
            Re = float(Re0 * np.exp(rng.normal(0.0, args.re_jitter)))
            if RE_LAM_MAX <= Re <= RE_TURB_MIN:
                continue
            f_clean, branch = moody_f(np.array([Re]), np.array([er]))
            f_clean, branch = float(f_clean[0]), str(branch[0])
            f_val = float(f_clean * np.exp(rng.normal(0.0, args.noise)))
            v = Re * mu0 / (rho0 * D0)
            dP_L = f_val * rho0 * v**2 / (2.0 * D0)
            rows.append({
                "filename": f"moody_eps{er:.2e}_Re{Re0:.2e}",
                "dP_L": dP_L, "v": v, "mu": mu0, "rho": rho0,
                "D": D0, "eps": er * D0,
                "f": f_val, "Re": Re, "eps_rel": er,
                "f_true": f_clean, "branch": branch,
            })
    syn = pd.DataFrame(rows)

    out = args.output if os.path.isabs(args.output) \
        else os.path.join(_here, args.output)
    syn.to_csv(out, index=False)

    n_lam = int((syn["branch"] == "laminar").sum())
    print(f"\nRows: {len(syn)}  ({n_lam} laminar, {len(syn)-n_lam} "
          f"turbulent;  f {syn['f'].min():.4f} .. {syn['f'].max():.4f})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
