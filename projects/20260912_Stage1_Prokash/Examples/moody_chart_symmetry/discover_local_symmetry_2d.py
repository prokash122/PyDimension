"""Region-free 2-D local symmetry discovery — the Moody chart test.

Extends the Windowed Local Symmetry Scan (WLSS) of
../porous_media_lbm_symmetry/discover_local_symmetry.py from 1-D interval
scans to a 2-D REGIME MAP.  Needed here because the Moody chart's
smooth-turbulent / fully-rough boundary is a diagonal in the
(Re, eps/D) plane — no single Pi-axis scan can resolve it.

Method
------
1. TILE     — partition the data into an nx x ny grid of quantile tiles
              in (ln Re, ln eps/D).  No regime information is used.
2. FIT      — per tile, OLS plane in log space:
                  ln f = ln C + a ln Re + b ln(eps/D)
              (a, b) is the local scaling-exponent vector = the local
              scaling-symmetry direction; its null space spans the local
              symmetry generators.
3. UQ       — pairs bootstrap per tile -> 95% percentile bands on a and
              b plus their bootstrap covariance.
4. VALIDITY — data-driven noise floor (10th percentile of tile residual
              RMS); tiles above 3x the floor contain regime structure
              at this resolution and are excluded from regimes
              (-> transition tiles), never fitted.
5. GROW     — region growing over the tile grid: seeds in order of
              residual RMS; a 4-neighbour tile joins a region if its
              exponents match the region's POOLED fit (chi^2 on
              bootstrap covariances OR all shifts < --merge-tol, the
              same ROPE rule as the 1-D scan).  Anchoring the test to
              the pooled fit prevents slow drift from chaining two
              different regimes together.
6. REPORT   — pooled law + bands per discovered region, regime map and
              Moody-chart figures, JSON, and per-regime CSVs for the
              existing Stage-1 tools.

Ground truth (never shown to the scan), for --compare-moody:
    laminar        : f = 64 Re^-1                (a,b) = (-1, 0)
    smooth turb.   : f = 0.316 Re^-1/4 (Blasius) (a,b) = (-0.25, 0)
    fully rough    : von Karman log-law          a = 0,
                     b_local = 2/ln(3.7/eps_rel) (~0.19..0.34 here)

Usage:
  python discover_local_symmetry_2d.py --data dataset_moody.csv \
      --output-dir output_moody --compare-moody
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHI2_2_95 = 5.991


# ----------------------------------------------------------------- data

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df = df[(df["f"] > 0) & (df["Re"] > 0) & (df["eps_rel"] > 0)]
    return df.reset_index(drop=True)


# --------------------------------------------------------- tile fitting

def fit_tile(X, y, idx, B, rng):
    """OLS plane ln f = lnC + a x1 + b x2 on rows idx, with pairs
    bootstrap bands and covariance of (a, b)."""
    n = len(idx)
    A = np.column_stack([X[idx], np.ones(n)])
    yw = y[idx]
    coef, *_ = np.linalg.lstsq(A, yw, rcond=None)
    g, lnC = coef[:2], coef[2]
    resid = yw - A @ coef
    rms = float(np.sqrt(np.mean(resid**2)))
    ss = float(np.sum((yw - yw.mean())**2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss if ss > 0 else float("nan")

    gs = np.empty((B, 2))
    for b in range(B):
        take = idx[rng.integers(0, n, n)]
        Ab = np.column_stack([X[take], np.ones(n)])
        cb, *_ = np.linalg.lstsq(Ab, y[take], rcond=None)
        gs[b] = cb[:2]
    lo, hi = np.percentile(gs, [2.5, 97.5], axis=0)
    cov = np.cov(gs.T) + 1e-12 * np.eye(2)
    return {"n": n, "g": g, "lnC": float(lnC), "rms": rms, "r2": r2,
            "ci_lo": lo, "ci_hi": hi, "cov": cov, "idx": idx}


def quantile_tiles(x1, x2, nx, ny):
    """(nx*ny) index sets from quantile bins along each coordinate."""
    def bins(v, k):
        edges = np.quantile(v, np.linspace(0, 1, k + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, k - 1)
    b1, b2 = bins(x1, nx), bins(x2, ny)
    tiles = {}
    for i in range(nx):
        for j in range(ny):
            idx = np.where((b1 == i) & (b2 == j))[0]
            if len(idx) >= 10:
                tiles[(i, j)] = idx
    return tiles


# -------------------------------------------------------- region growing

def mahalanobis2(g1, S1, g2, S2):
    d = g1 - g2
    return float(d @ np.linalg.solve(S1 + S2, d))


def grow_regions(tiles, fits, X, y, B, rng, noise_floor, merge_tol,
                 misfit_factor=3.0):
    """Grow statistically homogeneous regions over the tile grid."""
    valid = {t: f["rms"] <= misfit_factor * noise_floor
             for t, f in fits.items()}
    unassigned = {t for t, v in valid.items() if v}
    regions = []
    order = sorted(unassigned, key=lambda t: fits[t]["rms"])
    for seed in order:
        if seed not in unassigned:
            continue
        region = {"tiles": [seed], "idx": fits[seed]["idx"],
                  "pooled": fits[seed]}
        unassigned.discard(seed)
        frontier = [seed]
        while frontier:
            i, j = frontier.pop(0)
            for t in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if t not in unassigned:
                    continue
                ft, pool = fits[t], region["pooled"]
                d2 = mahalanobis2(pool["g"], pool["cov"], ft["g"], ft["cov"])
                equiv = np.max(np.abs(pool["g"] - ft["g"])) < merge_tol
                if d2 < CHI2_2_95 or equiv:
                    region["tiles"].append(t)
                    region["idx"] = np.union1d(region["idx"], ft["idx"])
                    region["pooled"] = fit_tile(X, y, region["idx"], B, rng)
                    unassigned.discard(t)
                    frontier.append(t)
        regions.append(region)

    regimes = [r for r in regions if len(r["tiles"]) >= 3]
    regimes.sort(key=lambda r: -len(r["tiles"]))
    leftover_tiles = [t for t in fits
                      if not any(t in r["tiles"] for r in regimes)]
    return regimes, leftover_tiles, valid


# -------------------------------------------------------------- helpers

def rough_b_local(eps_rel_mean):
    """Local eps/D exponent of the von Karman fully-rough law at the
    given mean relative roughness: d ln f / d ln eps = 2/ln(3.7/eps)."""
    return 2.0 / np.log(3.7 / eps_rel_mean)


def classify(g, eps_rel_mean):
    refs = {
        "laminar (f = 64/Re)": np.array([-1.0, 0.0]),
        "smooth turbulent (Blasius)": np.array([-0.25, 0.0]),
        "fully rough (von Karman)": np.array(
            [0.0, rough_b_local(eps_rel_mean)]),
    }
    best = min(refs, key=lambda k: np.linalg.norm(g - refs[k]))
    return best, refs

def fmt_arr(a, k=3):
    return "[" + ", ".join(f"{v:+.{k}f}" for v in a) + "]"


# ------------------------------------------------------------------ scan

def run_scan(df, out_dir, nx, ny, B, seed, merge_tol, compare_moody,
             emit_regimes):
    rng = np.random.default_rng(seed)
    Re, er, f = df["Re"].values, df["eps_rel"].values, df["f"].values
    X = np.column_stack([np.log(Re), np.log(er)])
    y = np.log(f)

    tiles = quantile_tiles(X[:, 0], X[:, 1], nx, ny)
    fits = {t: fit_tile(X, y, idx, B, rng) for t, idx in tiles.items()}
    noise_floor = float(np.quantile([f_["rms"] for f_ in fits.values()],
                                    0.10))
    regimes, leftover, valid = grow_regions(
        tiles, fits, X, y, B, rng, noise_floor, merge_tol)

    lines = []
    lines.append("REGION-FREE 2-D LOCAL SYMMETRY SCAN (WLSS-2D) — "
                 "Moody chart")
    lines.append(f"Dataset rows            : {len(y)}")
    lines.append(f"Tile grid               : {nx} x {ny} quantile tiles "
                 f"in (ln Re, ln eps/D)  "
                 f"(~{len(y)//(nx*ny)} rows/tile)")
    lines.append(f"Bootstrap resamples     : {B}")
    lines.append(f"Merge tolerance (ROPE)  : {merge_tol:.3f}")
    lines.append(f"Local model             : ln f = ln C + a ln Re "
                 f"+ b ln(eps/D)")
    lines.append(f"Noise floor (10th pct tile RMS) : {noise_floor:.4f} "
                 f"[ln f units]")
    n_valid = sum(valid.values())
    lines.append(f"Tiles at noise floor (<=3x)     : "
                 f"{n_valid}/{len(fits)}")
    lines.append(f"Discovered regimes (>=3 tiles)  : {len(regimes)}")
    lines.append("")

    results = {"n_rows": len(y), "grid": [nx, ny], "bootstrap": B,
               "merge_tol": merge_tol, "noise_floor": noise_floor,
               "regimes": []}
    for k, r in enumerate(regimes, 1):
        p = r["pooled"]
        idx = r["idx"]
        C = float(np.exp(p["lnC"]))
        re_lo, re_hi = np.log10(Re[idx].min()), np.log10(Re[idx].max())
        er_lo, er_hi = np.log10(er[idx].min()), np.log10(er[idx].max())
        er_mean = float(np.exp(np.mean(np.log(er[idx]))))
        tile_g = np.array([fits[t]["g"] for t in r["tiles"]])
        inhom = float(np.max(np.abs(tile_g - p["g"])))
        lines.append("-" * 72)
        lines.append(f"REGIME {k}  ({len(r['tiles'])} tiles, {p['n']} rows)")
        lines.append(f"  span              : log10 Re in "
                     f"[{re_lo:.2f}, {re_hi:.2f}],  log10 eps/D in "
                     f"[{er_lo:.2f}, {er_hi:.2f}]")
        lines.append(f"  exponents (a, b)  : {fmt_arr(p['g'])}")
        lines.append(f"  95% bands         : a [{p['ci_lo'][0]:+.3f},"
                     f"{p['ci_hi'][0]:+.3f}]   b [{p['ci_lo'][1]:+.3f},"
                     f"{p['ci_hi'][1]:+.3f}]")
        lines.append(f"  local law         : f = {C:.4g} "
                     f"* Re^{p['g'][0]:+.3f} * (eps/D)^{p['g'][1]:+.3f}")
        lines.append(f"  pooled R2 (ln f)  : {p['r2']:.4f}   residual "
                     f"RMS = {p['rms']:.4f}")
        lines.append(f"  tile-vs-pooled max exponent dev : {inhom:.3f}")
        reg = {"n_tiles": len(r["tiles"]), "n_rows": int(p["n"]),
               "span_log10_Re": [re_lo, re_hi],
               "span_log10_eps_rel": [er_lo, er_hi],
               "exponents": p["g"].tolist(),
               "ci_lo": p["ci_lo"].tolist(), "ci_hi": p["ci_hi"].tolist(),
               "prefactor_C": C, "r2_lnf": p["r2"], "rms": p["rms"],
               "max_tile_dev": inhom}
        if compare_moody:
            name, refs = classify(p["g"], er_mean)
            ref = refs[name]
            lines.append(f"  --> best match    : {name}")
            lines.append(f"      reference (a, b) = {fmt_arr(ref)}"
                         + ("   [b_local = 2/ln(3.7/eps_mean), "
                            f"eps_mean = {er_mean:.2e}]"
                            if "rough" in name else ""))
            lines.append(f"      deviation        = "
                         f"{np.linalg.norm(p['g'] - ref):.3f}")
            reg["best_match"] = name
            reg["reference_exponents"] = ref.tolist()
        results["regimes"].append(reg)
    lines.append("-" * 72)
    n_trans = len(leftover)
    lines.append(f"Transition / unresolved tiles   : {n_trans} "
                 f"(regime boundaries: laminar-turbulent critical zone "
                 f"and the smooth-rough diagonal)")
    lines.append("")
    lines.append(f"VERDICT: {len(regimes)} local scaling symmetries "
                 f"discovered on the (Re, eps/D) plane without any "
                 f"region specification.")

    emitted = []
    if emit_regimes:
        for stale in out_dir.glob("regime_*_rows.csv"):
            stale.unlink()
        for k, r in enumerate(regimes, 1):
            sub = df.iloc[np.sort(r["idx"])]
            p = out_dir / f"regime_{k}_rows.csv"
            sub.to_csv(p, index=False)
            emitted.append(p.name)
            lines.append(f"  regime {k} rows -> {p.name} ({len(sub)} rows)")
    results["emitted_regime_files"] = emitted

    report = "\n".join(lines) + "\n"
    (out_dir / "local_symmetry_2d_report.txt").write_text(report)
    with open(out_dir / "local_symmetry_2d_scan.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(report)
    return tiles, fits, regimes, leftover, valid, noise_floor


# -------------------------------------------------------------- plotting

REGIME_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd"]


def plot_regime_map(df, tiles, fits, regimes, valid, out_dir):
    Re, er = df["Re"].values, df["eps_rel"].values
    x1, x2 = np.log10(Re), np.log10(er)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

    tile_of = {}
    for k, r in enumerate(regimes):
        for t in r["tiles"]:
            tile_of[t] = k

    def tile_patch(ax, t, color, alpha):
        idx = tiles[t]
        ax.fill([x1[idx].min(), x1[idx].max(), x1[idx].max(),
                 x1[idx].min()],
                [x2[idx].min(), x2[idx].min(), x2[idx].max(),
                 x2[idx].max()], color=color, alpha=alpha, lw=0)

    for pnl, comp in ((0, 0), (1, 1)):
        ax = axes[pnl]
        vals = {t: fits[t]["g"][comp] for t in tiles}
        vmin, vmax = min(vals.values()), max(vals.values())
        for t in tiles:
            frac = (vals[t] - vmin) / (vmax - vmin + 1e-12)
            ax.scatter(np.mean(x1[tiles[t]]), np.mean(x2[tiles[t]]),
                       c=[plt.cm.coolwarm(frac)], s=250, marker="s")
            if not valid[t]:
                ax.scatter(np.mean(x1[tiles[t]]), np.mean(x2[tiles[t]]),
                           marker="x", c="k", s=60)
        ax.set_title(f"local exponent {'a (Re)' if comp == 0 else 'b (eps/D)'}"
                     f"  [{vmin:+.2f} .. {vmax:+.2f}]  (x = misfit tile)")
        ax.set_xlabel("log10 Re")
    axes[0].set_ylabel("log10 eps/D")

    ax = axes[2]
    for t in tiles:
        if t in tile_of:
            tile_patch(ax, t, REGIME_COLORS[tile_of[t] % 5], 0.55)
        else:
            tile_patch(ax, t, "0.4", 0.35)
    for k, r in enumerate(regimes):
        idx = r["idx"]
        ax.text(np.median(x1[idx]), np.median(x2[idx]), f"R{k+1}",
                ha="center", va="center", fontsize=13, weight="bold")
    ax.set_title("discovered regime map (grey = transition)")
    ax.set_xlabel("log10 Re")
    fig.suptitle("2-D local symmetry scan on the Moody chart — "
                 "no region was specified")
    fig.tight_layout()
    fig.savefig(out_dir / "moody_regime_map.png", dpi=140)
    plt.close(fig)


def plot_moody_chart(df, regimes, out_dir):
    from generate_moody_dataset import colebrook_f
    Re, er, f = df["Re"].values, df["eps_rel"].values, df["f"].values
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(Re, f, ".", ms=2, color="lightgray", label="all rows")
    for k, r in enumerate(regimes):
        idx = r["idx"]
        ax.loglog(Re[idx], f[idx], ".", ms=2.5,
                  color=REGIME_COLORS[k % 5],
                  label=f"regime {k+1}: "
                        f"f ~ Re^{r['pooled']['g'][0]:+.2f} "
                        f"(eps/D)^{r['pooled']['g'][1]:+.2f}")
    Re_lam = np.logspace(np.log10(300), np.log10(2300), 50)
    ax.loglog(Re_lam, 64 / Re_lam, "k--", lw=1)
    Re_t = np.logspace(np.log10(4000), 8, 120)
    for e0 in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2]:
        ax.loglog(Re_t, colebrook_f(Re_t, np.full_like(Re_t, e0)),
                  "k--", lw=0.6)
    ax.set_xlabel("Re")
    ax.set_ylabel("Darcy friction factor f")
    ax.set_title("Moody chart coloured by discovered regime "
                 "(dashed = ground-truth curves)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "moody_chart_discovered.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--output-dir", default="output_moody")
    ap.add_argument("--nx", type=int, default=14,
                    help="tile columns along ln Re (default 14)")
    ap.add_argument("--ny", type=int, default=6,
                    help="tile rows along ln eps/D (default 6)")
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge-tol", type=float, default=0.10)
    ap.add_argument("--compare-moody", action="store_true",
                    help="report deviations vs the known laminar/Blasius/"
                         "von-Karman exponents (synthetic validation)")
    ap.add_argument("--no-emit-regimes", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = args.data if os.path.exists(args.data) else str(here / args.data)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = here / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    tiles, fits, regimes, leftover, valid, floor = run_scan(
        df, out_dir, args.nx, args.ny, args.bootstrap, args.seed,
        args.merge_tol, args.compare_moody, not args.no_emit_regimes)

    plot_regime_map(df, tiles, fits, regimes, valid, out_dir)
    plot_moody_chart(df, regimes, out_dir)
    print(f"\nWrote report, JSON, and figures to {out_dir}")


if __name__ == "__main__":
    main()
