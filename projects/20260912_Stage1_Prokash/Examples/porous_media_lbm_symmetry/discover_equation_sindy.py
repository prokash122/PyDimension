"""
SINDy-style post-processor: turn the Stage1 pipeline's discovered
symmetry direction into an explicit power-law equation with integer-like
exponents and a fitted prefactor.

The Stage1 Step-3 scaling encoder found the right latent variable and
the right symmetry type (scaling), but its individual exponents on
`phi` and `(1-phi)` sit in an unidentified member of a 1-parameter
equivalence class -- the flexible neural decoder absorbs any residual
curvature.  This script sidesteps that by doing a plain Lasso
regression on

    log f  =  a . log(Re_p)  +  b . log(phi)  +  c . log(1-phi)  +  const

with a tiny L1 penalty so small-effect coefficients snap to exactly
zero (a-la SINDy).  The linear-decoder assumption is honest here because
the pipeline already told us the underlying law is a single monomial
(k*=1, scaling symmetry).

Run it region by region:
    python discover_equation_sindy.py \
        --data dataset_lbm_porous.csv --out output_viscous_region
    python discover_equation_sindy.py \
        --data dataset_ergun_inertial_widephi.csv --out output_inertial_widephi
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV, LinearRegression

_here = os.path.dirname(os.path.abspath(__file__))


def load_encoder_direction(run_log_path):
    """Parse the Stage1 encoder's L2-normed direction from run.log.

    Looks for the line printed as
        L2-n:    -0.2857  -0.5356  +0.7947
    and returns the three floats as an ndarray plus the manifold-cos line
    if present.  Returns None if the log doesn't have it (older format /
    no scaling winner).
    """
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
    # only positive quantities are safe for log
    d = d[(d["f"] > 0) & (d["v"] > 0) & (d["Re_p"] > 0) &
          (d["phi"] > 0) & (d["phi"] < 1)]
    return d


def round_near_integer(x, tol=0.15):
    """Snap x to nearest integer if within tol, else leave as fitted."""
    r = round(x)
    return r if abs(x - r) < tol else x


def fmt_exp(v):
    """Format an exponent, showing 0 as '0'."""
    if abs(v) < 1e-3:
        return "0"
    return f"{v:+.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True,
                    help="Directory to write sindy_equation.txt into.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Explicit Lasso alpha; if omitted, use LassoCV.")
    args = ap.parse_args()

    path = args.data if os.path.exists(args.data) else \
        os.path.join(_here, args.data)
    df = read_ok(path)
    Re = df["Re_p"].to_numpy()
    phi = df["phi"].to_numpy()
    f = df["f"].to_numpy()

    logf = np.log(f)
    X = np.column_stack([np.log(Re), np.log(phi), np.log(1 - phi)])
    names = ["log(Re_p)", "log(phi)", "log(1-phi)"]

    print(f"Data source        : {os.path.basename(path)}   (n = {len(df)})")

    # ---- 0) Load Stage1 encoder direction from run.log -------------------
    enc = load_encoder_direction(os.path.join(args.out, "run.log"))
    if enc is not None:
        w_enc, mcos = enc
        # Rescale so the largest-|weight| component = 1, for readability
        big = int(np.argmax(np.abs(w_enc)))
        w_enc_scaled = w_enc / w_enc[big]
        print(f"Stage1 encoder w   : "
              f"[{', '.join(f'{c:+.3f}' for c in w_enc)}]  "
              f"(L2-normed, from run.log)")
        print(f"  rescaled (large=1): "
              f"[{', '.join(f'{c:+.3f}' for c in w_enc_scaled)}]"
              f"   manifold cos = {mcos:+.4f}")
    else:
        w_enc = None
        print("(no Stage1 encoder direction found; SINDy runs standalone)")

    # ---- 1) plain OLS baseline (no regularisation) ------------------------
    ols = LinearRegression().fit(X, logf)
    print(f"OLS (no penalty)   : ", end="")
    print(f"[{', '.join(f'{c:+.3f}' for c in ols.coef_)}]  "
          f"const = {ols.intercept_:+.3f}   R2 = {ols.score(X, logf):.4f}")

    # ---- 2) LassoCV: pick alpha by 5-fold CV ------------------------------
    if args.alpha is None:
        lasso = LassoCV(cv=5, alphas=100, max_iter=200000,
                        fit_intercept=True, random_state=42).fit(X, logf)
        alpha = lasso.alpha_
    else:
        alpha = args.alpha
        lasso = Lasso(alpha=alpha, max_iter=200000,
                      fit_intercept=True).fit(X, logf)
    print(f"Lasso (alpha={alpha:.4g}) : ", end="")
    print(f"[{', '.join(f'{c:+.3f}' for c in lasso.coef_)}]  "
          f"const = {lasso.intercept_:+.3f}   R2 = {lasso.score(X, logf):.4f}")

    # ---- 3) snap-to-integer + refit intercept only ------------------------
    exps_snap = np.array([round_near_integer(c) for c in lasso.coef_])
    logC = float(np.mean(logf - X @ exps_snap))
    C = float(np.exp(logC))
    y_pred = np.exp(logC + X @ exps_snap)
    r2_snap = 1 - np.sum((f - y_pred) ** 2) / np.sum((f - f.mean()) ** 2)
    print(f"Snap-to-integer    : [{', '.join(str(int(v)) if v == int(v) else f'{v:+.2f}' for v in exps_snap)}]  "
          f"prefactor C = {C:.3g}   R2(f) = {r2_snap:.4f}")

    # ---- 3.5) Consistency check: does SINDy's direction agree with
    #           the Stage1 encoder direction? --------------------------
    if w_enc is not None:
        # Encoder direction is arbitrary in scale + sign; measure angle.
        v_sindy = np.array(lasso.coef_)
        if np.linalg.norm(v_sindy) > 0 and np.linalg.norm(w_enc) > 0:
            cos_raw = float(
                np.dot(w_enc, v_sindy) /
                (np.linalg.norm(w_enc) * np.linalg.norm(v_sindy)))
            # Also project both onto the (phi, 1-phi) data-manifold tangent
            # dlog(1-phi) = -phi_bar/(1-phi_bar) * dlog(phi).
            phi_bar = float(df["phi"].mean())
            slope = -phi_bar / (1.0 - phi_bar)
            proj = lambda v: np.array([v[0], v[1] + slope * v[2]])
            we, vs = proj(w_enc), proj(v_sindy)
            if np.linalg.norm(we) > 0 and np.linalg.norm(vs) > 0:
                cos_manifold = float(
                    np.dot(we, vs) /
                    (np.linalg.norm(we) * np.linalg.norm(vs)))
            else:
                cos_manifold = float("nan")
            print(f"cos<encoder, SINDy>  raw = {cos_raw:+.4f}   "
                  f"manifold-projected = {cos_manifold:+.4f}")

    # ---- 4) emit human-readable equation ---------------------------------
    parts = []
    if exps_snap[0] != 0:
        parts.append(f"Re_p^{int(exps_snap[0]) if exps_snap[0] == int(exps_snap[0]) else f'{exps_snap[0]:+.2f}'}")
    if exps_snap[1] != 0:
        parts.append(f"phi^{int(exps_snap[1]) if exps_snap[1] == int(exps_snap[1]) else f'{exps_snap[1]:+.2f}'}")
    if exps_snap[2] != 0:
        parts.append(f"(1-phi)^{int(exps_snap[2]) if exps_snap[2] == int(exps_snap[2]) else f'{exps_snap[2]:+.2f}'}")
    eq = f"f = {C:.3g} " + " . ".join(parts) if parts else f"f = {C:.3g}"
    print(f"\nDiscovered equation (SINDy post-process):")
    print(f"  {eq}")

    out_dir = args.out if os.path.isabs(args.out) else \
        os.path.join(_here, args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, "sindy_equation.txt")
    with open(out_txt, "w") as fh:
        fh.write(f"Data source: {os.path.basename(path)}   n = {len(df)}\n\n")
        if w_enc is not None:
            fh.write(f"Stage1 encoder w (L2)  : "
                     f"[{', '.join(f'{c:+.4f}' for c in w_enc)}]  "
                     f"manifold cos = {mcos:+.4f}\n")
            fh.write(f"  rescaled (large=1)   : "
                     f"[{', '.join(f'{c:+.4f}' for c in w_enc_scaled)}]\n")
        fh.write(f"OLS         : [{', '.join(f'{c:+.4f}' for c in ols.coef_)}]"
                 f"  const = {ols.intercept_:+.4f}  R2 = {ols.score(X, logf):.4f}\n")
        fh.write(f"LassoCV     : [{', '.join(f'{c:+.4f}' for c in lasso.coef_)}]"
                 f"  const = {lasso.intercept_:+.4f}  R2 = {lasso.score(X, logf):.4f}"
                 f"  alpha = {alpha:.4g}\n")
        fh.write(f"Snap-to-int : [{', '.join(str(int(v)) if v == int(v) else f'{v:+.4f}' for v in exps_snap)}]"
                 f"  prefactor C = {C:.4g}  R2(f) = {r2_snap:.4f}\n\n")
        if w_enc is not None:
            fh.write(f"cos<encoder, SINDy>  raw = {cos_raw:+.4f}   "
                     f"manifold-projected = {cos_manifold:+.4f}\n\n")
        fh.write(f"Discovered equation:  {eq}\n")
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
