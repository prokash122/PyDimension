"""
Extract the discovered equation from the Stage1 pipeline output using
proper L2 norm (ordinary least squares) — no L1 thresholding, no
Lasso penalty.

Given that Stage1's Step 3 already told us:
  1. The symmetry type is **scaling** (winner encoder), and
  2. The intrinsic latent dimension k* = 1,

we know f is a *single monomial* of the Pi variables:
    f = C · Re_p^a · phi^b · (1 - phi)^c
Taking log turns this into a plain linear model
    log f  =  a·log Re_p  +  b·log phi  +  c·log(1 - phi)  +  log C
which is fit by **ordinary least squares** — the pure L2-norm
minimisation:
    (a*, b*, c*, log C*)  =  argmin  Σ ( log f_i  -  a·log Re_i
                                        -  b·log phi_i
                                        -  c·log(1 - phi_i)
                                        -  log C )^2

The encoder direction from Stage1 is read from `run.log` and reported
alongside the OLS answer as a *consistency check* (cos-angle in raw
3-D and along the (phi, 1-phi) data-manifold tangent); it is NOT used
as a constraint in the fit.

Then near-integer OLS coefficients are snapped to integers and the
prefactor is re-fit by matching medians.

Usage:
    python discover_equation_l2.py --data <csv> --out <output-dir>

The output-dir must already contain the pipeline's `run.log` if you
want the encoder consistency check to run.
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

_here = os.path.dirname(os.path.abspath(__file__))


def load_encoder_direction(run_log_path):
    if not os.path.exists(run_log_path):
        return None
    with open(run_log_path) as fh:
        text = fh.read()
    m = re.search(r"L2-n:\s*([+\-\d.eE]+)\s+([+\-\d.eE]+)\s+([+\-\d.eE]+)", text)
    if not m:
        return None
    w = np.array([float(m.group(i)) for i in (1, 2, 3)])
    mos = re.search(r"cos along data manifold[^:]*:\s*([+\-\d.eE]+)", text)
    manifold_cos = float(mos.group(1)) if mos else None
    return w, manifold_cos


def read_ok(path):
    df = pd.read_csv(path)
    ok = (df["converged"].astype(str).str.lower() == "true") & \
         (df["stalled"].astype(str).str.lower() != "true")
    d = df[ok].copy()
    d = d[(d["f"] > 0) & (d["v"] > 0) & (d["Re_p"] > 0) &
          (d["phi"] > 0) & (d["phi"] < 1)]
    return d


def round_near_integer(x, tol=0.15):
    r = round(x)
    return r if abs(x - r) < tol else x


def format_expr(exps, C):
    """Return an f = C · Re^a · phi^b · (1-phi)^c string, skipping ^0 terms."""
    parts = []
    labels = ("Re_p", "phi", "(1-phi)")
    for e, lbl in zip(exps, labels):
        if abs(e) < 1e-3:
            continue
        if e == int(e):
            parts.append(f"{lbl}^{int(e)}")
        else:
            parts.append(f"{lbl}^{e:+.2f}")
    return f"f = {C:.3g}" + (" · " + " · ".join(parts) if parts else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = args.data if os.path.exists(args.data) else \
        os.path.join(_here, args.data)
    df = read_ok(path)
    Re = df["Re_p"].to_numpy()
    phi = df["phi"].to_numpy()
    f = df["f"].to_numpy()

    logf = np.log(f)
    X = np.column_stack([np.log(Re), np.log(phi), np.log(1 - phi)])

    print(f"Data source        : {os.path.basename(path)}   (n = {len(df)})")

    # --- Stage1 encoder direction (for consistency check only) -----------
    enc = load_encoder_direction(os.path.join(args.out, "run.log"))
    if enc is not None:
        w_enc, mcos = enc
        print(f"Stage1 encoder w   : "
              f"[{', '.join(f'{c:+.4f}' for c in w_enc)}]  "
              f"(L2-normed, from run.log; manifold cos = {mcos:+.4f})")

    # --- OLS: pure L2 fit on log-features --------------------------------
    ols = LinearRegression().fit(X, logf)
    exps = ols.coef_
    r2_logf = ols.score(X, logf)
    print(f"OLS (L2 fit)       : "
          f"[{', '.join(f'{c:+.4f}' for c in exps)}]  "
          f"const = {ols.intercept_:+.4f}   R2(log f) = {r2_logf:.4f}")

    # --- Snap near-integer exponents to integers, refit prefactor --------
    exps_snap = np.array([round_near_integer(v) for v in exps])
    logC = float(np.mean(logf - X @ exps_snap))
    C = float(np.exp(logC))
    y_pred = np.exp(logC + X @ exps_snap)
    r2_f = 1 - np.sum((f - y_pred) ** 2) / np.sum((f - f.mean()) ** 2)
    print(f"Snap-to-integer    : "
          f"[{', '.join(str(int(v)) if v == int(v) else f'{v:+.2f}' for v in exps_snap)}]   "
          f"C = {C:.4g}   R2(f) = {r2_f:.4f}")

    # --- Encoder consistency check ---------------------------------------
    if enc is not None:
        v_ols = np.array(exps)
        cos_raw = float(np.dot(w_enc, v_ols) /
                        (np.linalg.norm(w_enc) * np.linalg.norm(v_ols)))
        phi_bar = float(df["phi"].mean())
        slope = -phi_bar / (1.0 - phi_bar)
        proj = lambda v: np.array([v[0], v[1] + slope * v[2]])
        cos_manifold = float(
            np.dot(proj(w_enc), proj(v_ols)) /
            (np.linalg.norm(proj(w_enc)) * np.linalg.norm(proj(v_ols))))
        print(f"cos<encoder, OLS>  raw = {cos_raw:+.4f}   "
              f"manifold-projected = {cos_manifold:+.4f}")

    eq = format_expr(exps_snap, C)
    print(f"\nDiscovered equation (proper L2 fit):")
    print(f"  {eq}")

    out_dir = args.out if os.path.isabs(args.out) else \
        os.path.join(_here, args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, "l2_equation.txt")
    with open(out_txt, "w") as fh:
        fh.write(f"Data source: {os.path.basename(path)}   n = {len(df)}\n\n")
        if enc is not None:
            fh.write(f"Stage1 encoder w  : "
                     f"[{', '.join(f'{c:+.4f}' for c in w_enc)}]  "
                     f"manifold cos = {mcos:+.4f}\n")
        fh.write(f"OLS (L2 fit)       : "
                 f"[{', '.join(f'{c:+.4f}' for c in exps)}]  "
                 f"const = {ols.intercept_:+.4f}  R2(log f) = {r2_logf:.4f}\n")
        fh.write(f"Snap-to-integer   : "
                 f"[{', '.join(str(int(v)) if v == int(v) else f'{v:+.4f}' for v in exps_snap)}]  "
                 f"C = {C:.4g}  R2(f) = {r2_f:.4f}\n")
        if enc is not None:
            fh.write(f"cos<encoder, OLS>: raw = {cos_raw:+.4f}   "
                     f"manifold = {cos_manifold:+.4f}\n")
        fh.write(f"\nDiscovered equation:  {eq}\n")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
