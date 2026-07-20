"""Region-free LOCAL symmetry discovery with bootstrap uncertainty bands.

Answers: "can we discover the local scaling symmetries of a dataset that
spans several physical regimes WITHOUT telling the pipeline where the
regimes are?"

Method — Windowed Local Symmetry Scan (WLSS)
--------------------------------------------

The Stage-1 result for each pure Ergun branch is that `k* = 1` with a
SCALING symmetry, i.e. locally

    f  =  C * Re_p^a * phi^b * (1-phi)^c        (a single monomial)

In log-Pi coordinates x = [ln Re_p, ln phi, ln(1-phi)] a monomial is a
plane: ln f = ln C + g . x, and the exponent vector g = (a, b, c) IS the
local scaling-symmetry direction (its 2-D null space are the local
generators).  So the local symmetry can be estimated per data window by
local linear regression in log space — no region labels needed:

1. SCAN     — slide overlapping quantile windows along every log-Pi
              coordinate in turn (no axis is privileged a priori).
2. FIT      — in each window, OLS of ln f on [ln Re, ln phi, ln(1-phi)]
              gives the local exponent vector g_w and prefactor.
3. UQ       — pairs bootstrap (resample rows within the window) gives
              percentile uncertainty BANDS on each exponent and the full
              bootstrap covariance of the direction.
4. VALIDITY — a window supports a local power law only if its residual
              RMS sits at the dataset noise floor (estimated data-driven
              as a low quantile of all window RMS values).  Windows far
              above the floor straddle a regime change at this width.
5. SEGMENT  — along each scan axis, merge consecutive NON-overlapping
              windows whose exponent vectors are statistically
              indistinguishable (chi^2 test on the identifiable,
              manifold-projected 2-D exponents using bootstrap
              covariances) OR practically equivalent (all projected
              exponent differences below --merge-tol, a region-of-
              practical-equivalence rule: with very low noise the bands
              become so tight that physically negligible drift is
              statistically detectable).  Maximal merged runs =
              discovered regimes; unmerged / misfit windows =
              transition zones.
6. CHECK    — the whole scan is repeated at twice the window width; a
              regime is only trusted where the coarse and fine estimates
              agree within the bands (multiscale stability).

Identifiability: phi and (1-phi) never vary independently (the data lie
on the manifold Pi4 = 1 - Pi3), so only the manifold-projected 2-D
exponents (a, b_eff) with b_eff = b + slope*c, slope = -phibar/(1-phibar)
are identifiable per window.  Segmentation tests are therefore run in
that projected space; the raw 3-D exponents are reported with their
(honestly wide, when ill-conditioned) bootstrap bands.

Outputs (per run, into --output-dir):
  local_symmetry_report.txt      human-readable report
  local_symmetry_scan.json       machine-readable everything
  local_symmetry_scan_<axis>.png exponent trajectories + bands + regimes
  local_symmetry_regimes.png     Ergun collapse coloured by discovered regime
  regime_<k>_rows.csv            per-regime dataset in the standard schema,
                                 ready for discover_symmetry.py /
                                 discover_equation_encoder_l2.py

Usage:
  python discover_local_symmetry.py --data dataset_ergun_full_curve.csv \
      --output-dir output_local_symmetry --compare-ergun
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

AXIS_NAMES = ["log10(Re_p)", "log10(phi)", "log10(1-phi)"]
EXP_NAMES = ["a (Re_p)", "b (phi)", "c (1-phi)"]
ERGUN_BRANCHES = {
    "viscous (Darcy)":          {"exps": np.array([-1.0, -3.0, 2.0]), "C": 150.0},
    "inertial (Forchheimer)":   {"exps": np.array([0.0, -3.0, 1.0]),  "C": 1.75},
}
CHI2_2_95 = 5.991  # chi^2 0.95 quantile, 2 dof (projected exponent space)


# ----------------------------------------------------------------- data

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "converged" in df.columns:
        conv = df["converged"].astype(str).str.lower() == "true"
        stall = df.get("stalled", "false").astype(str).str.lower() == "true"
        df = df[conv & ~stall]
    df = df[(df["f"] > 0) & (df["Re_p"] > 0)
            & (df["phi"] > 0) & (df["phi"] < 1)].reset_index(drop=True)
    return df


# ------------------------------------------------------- window fitting

def fit_window(X, y, phi, idx, B, rng):
    """OLS of y on X[idx] (+intercept) with pairs bootstrap.

    Returns exponents g=(a,b,c), lnC, residual RMS, R2, conditioning,
    percentile bands, and the bootstrap covariance of the
    manifold-projected 2-D exponents (a, b_eff)."""
    n = len(idx)
    A = np.column_stack([X[idx], np.ones(n)])
    yw = y[idx]
    coef, *_ = np.linalg.lstsq(A, yw, rcond=None)
    g, lnC = coef[:3], coef[3]
    resid = yw - A @ coef
    rms = float(np.sqrt(np.mean(resid**2)))
    ss = float(np.sum((yw - yw.mean())**2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss if ss > 0 else float("nan")
    Xc = X[idx] - X[idx].mean(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    cond = float(sv[0] / max(sv[-1], 1e-300))

    phibar = float(np.mean(phi[idx]))
    slope = -phibar / (1.0 - phibar)
    proj = lambda v: np.array([v[..., 0], v[..., 1] + slope * v[..., 2]]).T

    gs = np.empty((B, 3))
    for b in range(B):
        take = idx[rng.integers(0, n, n)]
        Ab = np.column_stack([X[take], np.ones(n)])
        cb, *_ = np.linalg.lstsq(Ab, y[take], rcond=None)
        gs[b] = cb[:3]
    lo, hi = np.percentile(gs, [2.5, 97.5], axis=0)
    gp = proj(gs)
    cov_proj = np.cov(gp.T) + 1e-12 * np.eye(2)

    return {
        "n": n, "g": g, "lnC": float(lnC), "rms": rms, "r2": r2,
        "cond": cond, "phibar": phibar, "slope": slope,
        "g_proj": proj(g[None, :])[0], "ci_lo": lo, "ci_hi": hi,
        "cov_proj": cov_proj, "idx": idx,
    }


def quantile_windows(coord, frac, centers):
    """Index sets of points whose quantile rank along `coord` lies within
    frac/2 of each center."""
    order = np.argsort(coord, kind="stable")
    q = np.empty(len(coord))
    q[order] = np.arange(len(coord)) / max(len(coord) - 1, 1)
    out = []
    for c in centers:
        idx = np.where(np.abs(q - c) <= frac / 2)[0]
        out.append(idx)
    return out


# --------------------------------------------------------- segmentation

def mahalanobis2(g1, S1, g2, S2):
    d = g1 - g2
    S = S1 + S2
    return float(d @ np.linalg.solve(S, d))


def bridge_regimes(segments, X, y, phi, B, rng, noise_floor, merge_tol,
                   misfit_factor):
    """Merge regime segments separated only by valid-power-law windows
    when their pooled exponents are equivalent (chi^2 or ROPE), and the
    merged pooled fit stays within merge_tol of both ends."""
    changed = True
    while changed:
        changed = False
        for i, si in enumerate(segments):
            if si["type"] != "regime":
                continue
            j = i + 1
            while j < len(segments) and segments[j]["type"] != "regime":
                j += 1
            if j >= len(segments):
                break
            between = segments[i + 1:j]
            if any(w["rms"] > misfit_factor * noise_floor
                   for s in between for w in s["windows"]):
                continue
            gi, gj = si["pooled"], segments[j]["pooled"]
            d2 = mahalanobis2(gi["g_proj"], gi["cov_proj"],
                              gj["g_proj"], gj["cov_proj"])
            equiv = np.max(np.abs(gi["g_proj"] - gj["g_proj"])) < merge_tol
            if not (d2 < CHI2_2_95 or equiv):
                continue
            idx = si["idx"]
            cs, ws = list(si["centers"]), list(si["windows"])
            for s in between + [segments[j]]:
                idx = np.union1d(idx, s["idx"])
                cs += s["centers"]
                ws += s["windows"]
            pooled = fit_window(X, y, phi, idx, B, rng)
            if (np.max(np.abs(pooled["g_proj"] - gi["g_proj"])) < merge_tol
                    and np.max(np.abs(pooled["g_proj"]
                                      - gj["g_proj"])) < merge_tol):
                segments[i:j + 1] = [{"type": "regime", "centers": cs,
                                      "windows": ws, "idx": idx,
                                      "pooled": pooled}]
                changed = True
                break
    return segments


def segment_axis(X, y, phi, coord, frac, B, rng, noise_floor,
                 merge_tol, misfit_factor=3.0):
    """Greedy merge of consecutive non-overlapping windows along `coord`
    into statistically homogeneous regimes."""
    centers = list(np.arange(frac / 2, 1 - frac / 2 + 1e-9, frac))
    wins = [fit_window(X, y, phi, idx, B, rng)
            for idx in quantile_windows(coord, frac, centers)]

    segments, current = [], None
    for c, w in zip(centers, wins):
        misfit = w["rms"] > misfit_factor * noise_floor
        if misfit:
            if current:
                segments.append(current)
            segments.append({"type": "transition", "centers": [c],
                             "windows": [w], "idx": w["idx"]})
            current = None
            continue
        if current is None:
            current = {"type": "regime", "centers": [c], "windows": [w],
                       "idx": w["idx"], "pooled": w}
            continue
        d2 = mahalanobis2(current["pooled"]["g_proj"],
                          current["pooled"]["cov_proj"],
                          w["g_proj"], w["cov_proj"])
        # merge if statistically indistinguishable OR practically
        # equivalent (all identifiable exponent shifts below merge_tol)
        equiv = np.max(np.abs(current["pooled"]["g_proj"]
                              - w["g_proj"])) < merge_tol
        if d2 < CHI2_2_95 or equiv:
            current["centers"].append(c)
            current["windows"].append(w)
            current["idx"] = np.union1d(current["idx"], w["idx"])
            current["pooled"] = fit_window(X, y, phi, current["idx"], B, rng)
        else:
            segments.append(current)
            current = {"type": "regime", "centers": [c], "windows": [w],
                       "idx": w["idx"], "pooled": w}
    if current:
        segments.append(current)

    # a "regime" needs at least two consistent windows; lone survivors are
    # reclassified as transition
    for s in segments:
        if s["type"] == "regime" and len(s["centers"]) < 2:
            s["type"] = "transition"

    # bridging post-pass: a single noisy window can falsely split one
    # physical regime.  If two regime segments are separated only by
    # windows that individually satisfy the power-law validity cut, and
    # their pooled exponents are statistically or practically equivalent,
    # merge everything between them into one regime.
    segments = bridge_regimes(segments, X, y, phi, B, rng,
                              noise_floor, merge_tol, misfit_factor)

    # merge contiguous transition windows into single zones
    merged = []
    for s in segments:
        if (merged and s["type"] == "transition"
                and merged[-1]["type"] == "transition"):
            merged[-1]["centers"] += s["centers"]
            merged[-1]["windows"] += s["windows"]
            merged[-1]["idx"] = np.union1d(merged[-1]["idx"], s["idx"])
        else:
            merged.append(s)
    return merged, wins, centers


# -------------------------------------------------------------- helpers

def cos_to(g, target, phibar):
    gn = g / (np.linalg.norm(g) + 1e-300)
    tn = target / np.linalg.norm(target)
    raw = float(np.dot(gn, tn))
    slope = -phibar / (1.0 - phibar)
    proj = lambda v: np.array([v[0], v[1] + slope * v[2]])
    gp, tp = proj(gn), proj(tn)
    man = float(gp @ tp / (np.linalg.norm(gp) * np.linalg.norm(tp) + 1e-300))
    return raw, man


def fmt_arr(a, k=3):
    return "[" + ", ".join(f"{v:+.{k}f}" for v in a) + "]"


# ------------------------------------------------------------ the scan

def run_scan(df, out_dir, frac, n_centers, B, seed, compare_ergun,
             emit_regimes, merge_tol):
    rng = np.random.default_rng(seed)
    Re, phi, f = df["Re_p"].values, df["phi"].values, df["f"].values
    X = np.column_stack([np.log(Re), np.log(phi), np.log(1 - phi)])
    y = np.log(f)

    centers_overlap = np.linspace(frac / 2, 1 - frac / 2, n_centers)
    results = {"axes": {}, "window_frac": frac, "n_rows": len(y),
               "seed": seed, "bootstrap": B}
    lines = []
    lines.append("REGION-FREE LOCAL SYMMETRY SCAN (WLSS)")
    lines.append(f"Dataset rows            : {len(y)}")
    lines.append(f"Window fraction         : {frac:.2f} of data "
                 f"(~{int(frac*len(y))} rows/window)")
    lines.append(f"Bootstrap resamples     : {B}")
    lines.append(f"Merge tolerance (ROPE)  : {merge_tol:.3f} on projected exponents")
    lines.append(f"Local model             : ln f = ln C + a ln Re_p "
                 f"+ b ln phi + c ln(1-phi)")
    lines.append("")

    # ---- data-driven noise floor: low quantile of all window RMS values
    all_rms = []
    for ax in range(3):
        for idx in quantile_windows(X[:, ax], frac, centers_overlap):
            A = np.column_stack([X[idx], np.ones(len(idx))])
            coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
            all_rms.append(float(np.sqrt(np.mean((y[idx] - A @ coef)**2))))
    noise_floor = float(np.quantile(all_rms, 0.10))
    results["noise_floor"] = noise_floor
    lines.append(f"Noise floor (10th pct of window residual RMS) : "
                 f"{noise_floor:.4f}  [ln f units]")
    lines.append("")

    axis_data = {}
    for ax in range(3):
        coord = X[:, ax]
        # overlapping trajectory at frac and 2*frac (multiscale check)
        traj, traj2 = [], []
        for idx in quantile_windows(coord, frac, centers_overlap):
            traj.append(fit_window(X, y, phi, idx, B, rng))
        for idx in quantile_windows(coord, min(2 * frac, 0.9),
                                    centers_overlap):
            A = np.column_stack([X[idx], np.ones(len(idx))])
            coef, *_ = np.linalg.lstsq(A, y[idx], rcond=None)
            traj2.append(coef[:3])
        traj2 = np.array(traj2)

        segments, seg_wins, seg_centers = segment_axis(
            X, y, phi, coord, frac, B, rng, noise_floor, merge_tol)
        n_resolving = sum(w["rms"] <= 3.0 * noise_floor for w in traj)
        resolving_frac = n_resolving / len(traj)

        # multiscale agreement inside the fine bands
        inband = [
            np.all((traj2[i] >= t["ci_lo"]) & (traj2[i] <= t["ci_hi"]))
            for i, t in enumerate(traj)
        ]
        axis_data[ax] = {
            "traj": traj, "traj2": traj2, "centers": centers_overlap,
            "segments": segments, "seg_centers": seg_centers,
            "resolving_frac": resolving_frac, "inband": inband,
        }

        regimes = [s for s in segments if s["type"] == "regime"]
        lines.append("=" * 72)
        lines.append(f"SCAN AXIS: {AXIS_NAMES[ax]}")
        lines.append(f"  windows at noise floor (<=3x)   : "
                     f"{n_resolving}/{len(traj)}  "
                     f"({100*resolving_frac:.0f}% resolving)")
        lines.append(f"  multiscale (2x width in bands)  : "
                     f"{sum(inband)}/{len(inband)} centers")
        lines.append(f"  discovered regimes              : {len(regimes)}")

        ax_json = {"axis": AXIS_NAMES[ax],
                   "resolving_frac": resolving_frac,
                   "noise_floor": noise_floor, "regimes": [],
                   "transitions": []}
        for si, s in enumerate(segments):
            span_idx = s["idx"]
            lo10 = float(np.min(coord[span_idx]) / np.log(10))
            hi10 = float(np.max(coord[span_idx]) / np.log(10))
            if s["type"] != "regime":
                ax_json["transitions"].append(
                    {"span_log10": [lo10, hi10], "n_rows": len(span_idx)})
                continue
            p = s["pooled"]
            exps = p["g"]
            C = float(np.exp(p["lnC"]))
            lines.append("-" * 72)
            lines.append(f"  REGIME {len(ax_json['regimes'])+1}  "
                         f"[{AXIS_NAMES[ax]} in {lo10:+.2f} .. {hi10:+.2f}]  "
                         f"({len(s['centers'])} windows, {p['n']} rows)")
            lines.append(f"    exponents (a,b,c) : {fmt_arr(exps)}")
            lines.append(f"    95% bands         : a [{p['ci_lo'][0]:+.3f},"
                         f"{p['ci_hi'][0]:+.3f}]  b [{p['ci_lo'][1]:+.3f},"
                         f"{p['ci_hi'][1]:+.3f}]  c [{p['ci_lo'][2]:+.3f},"
                         f"{p['ci_hi'][2]:+.3f}]")
            lines.append(f"    local law         : f = {C:.4g} "
                         f"* Re_p^{exps[0]:+.3f} * phi^{exps[1]:+.3f} "
                         f"* (1-phi)^{exps[2]:+.3f}")
            lines.append(f"    window R2 (ln f)  : {p['r2']:.4f}   "
                         f"residual RMS = {p['rms']:.4f}")
            reg_json = {"span_log10": [lo10, hi10], "n_rows": int(p["n"]),
                        "exponents": exps.tolist(),
                        "ci_lo": p["ci_lo"].tolist(),
                        "ci_hi": p["ci_hi"].tolist(),
                        "prefactor_C": C, "r2_lnf": p["r2"],
                        "rms": p["rms"]}
            if compare_ergun:
                best = None
                for name, br in ERGUN_BRANCHES.items():
                    raw, man = cos_to(exps, br["exps"], p["phibar"])
                    lines.append(f"    cos vs Ergun {name:>22s} : "
                                 f"raw {raw:+.4f}   manifold {man:+.4f}")
                    if best is None or man > best[2]:
                        best = (name, raw, man)
                    reg_json[f"cos_{name.split()[0]}"] = {"raw": raw,
                                                          "manifold": man}
                lines.append(f"    --> best match: {best[0]} "
                             f"(manifold cos {best[2]:+.4f})")
                reg_json["best_match"] = best[0]
            ax_json["regimes"].append(reg_json)
        trans = ax_json["transitions"]
        if trans:
            spans = ", ".join(f"[{t['span_log10'][0]:+.2f}.."
                              f"{t['span_log10'][1]:+.2f}]" for t in trans)
            lines.append(f"  transition / no-local-power-law zones "
                         f"({AXIS_NAMES[ax]}): {spans}")
        lines.append("")
        results["axes"][AXIS_NAMES[ax]] = ax_json

    # ---- overall verdict: pick the most resolving axis with >1 regime
    best_ax = max(
        range(3),
        key=lambda a: (len([s for s in axis_data[a]["segments"]
                            if s["type"] == "regime"]),
                       axis_data[a]["resolving_frac"]),
    )
    regimes_best = [s for s in axis_data[best_ax]["segments"]
                    if s["type"] == "regime"]
    lines.append("=" * 72)
    lines.append(f"VERDICT: regime structure resolved along "
                 f"{AXIS_NAMES[best_ax]} — {len(regimes_best)} local "
                 f"scaling symmetries + transition zone(s) discovered "
                 f"without any region specification.")
    results["verdict_axis"] = AXIS_NAMES[best_ax]

    # ---- emit per-regime datasets for the existing Stage-1 pipeline
    emitted = []
    if emit_regimes:
        for stale in out_dir.glob("regime_*_rows.csv"):
            stale.unlink()
        for k, s in enumerate(regimes_best, 1):
            sub = df.iloc[np.sort(s["idx"])]
            p = out_dir / f"regime_{k}_rows.csv"
            sub.to_csv(p, index=False)
            emitted.append(str(p.name))
            lines.append(f"  regime {k} rows -> {p.name}  "
                         f"({len(sub)} rows; run discover_symmetry.py / "
                         f"discover_equation_encoder_l2.py on it to "
                         f"certify symmetry type per regime)")
    results["emitted_regime_files"] = emitted

    report = "\n".join(lines) + "\n"
    (out_dir / "local_symmetry_report.txt").write_text(report)
    with open(out_dir / "local_symmetry_scan.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(report)
    return axis_data, best_ax, results


# -------------------------------------------------------------- plotting

def plot_axis(axis_data, ax_i, out_dir, compare_ergun, noise_floor, X):
    d = axis_data[ax_i]
    Xax = X[:, ax_i]
    centers = d["centers"]
    traj, traj2 = d["traj"], d["traj2"]
    xc = np.array([np.median(Xax[t["idx"]]) for t in traj]) / np.log(10)
    G = np.array([t["g"] for t in traj])
    LO = np.array([t["ci_lo"] for t in traj])
    HI = np.array([t["ci_hi"] for t in traj])
    RMS = np.array([t["rms"] for t in traj])

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
    colors = ["#1f77b4", "#2ca02c", "#d62728"]
    truth = [(-1, 0), (-3, -3), (2, 1)] if compare_ergun else None
    for j in range(3):
        a = axes[j]
        a.fill_between(xc, LO[:, j], HI[:, j], alpha=0.25,
                       color=colors[j], label="95% bootstrap band")
        a.plot(xc, G[:, j], color=colors[j], lw=2,
               label="local exponent (window width w)")
        a.plot(xc, traj2[:, j], color="k", lw=1, ls=":",
               label="width 2w (multiscale check)")
        if truth:
            a.axhline(truth[j][0], color="gray", ls="--", lw=1)
            a.axhline(truth[j][1], color="gray", ls="-.", lw=1)
            a.text(1.005, truth[j][0], "viscous", transform=a.get_yaxis_transform(),
                   fontsize=7, va="center", color="gray")
            a.text(1.005, truth[j][1], "inertial", transform=a.get_yaxis_transform(),
                   fontsize=7, va="center", color="gray")
        for s in d["segments"]:
            if s["type"] == "regime":
                a.axvspan(np.min(Xax[s["idx"]]) / np.log(10),
                          np.max(Xax[s["idx"]]) / np.log(10),
                          color="green", alpha=0.06)
        a.set_ylabel(EXP_NAMES[j])
        if j == 0:
            a.legend(fontsize=8, loc="best")
    a = axes[3]
    a.semilogy(xc, RMS, "o-", ms=3, color="#9467bd")
    a.axhline(noise_floor, color="gray", ls="--", lw=1, label="noise floor")
    a.axhline(3 * noise_floor, color="red", ls=":", lw=1,
              label="3x floor (validity cut)")
    a.set_ylabel("window residual RMS")
    a.set_xlabel(AXIS_NAMES[ax_i])
    a.legend(fontsize=8)
    fig.suptitle(f"Local symmetry scan along {AXIS_NAMES[ax_i]} "
                 f"(green = discovered regimes)")
    fig.tight_layout()
    tag = AXIS_NAMES[ax_i].replace("log10(", "").replace(")", "") \
                          .replace("-", "m").replace("_", "")
    fig.savefig(out_dir / f"local_symmetry_scan_{tag}.png", dpi=140)
    plt.close(fig)


def plot_regimes_on_collapse(df, axis_data, best_ax, out_dir):
    Re, phi, f = df["Re_p"].values, df["phi"].values, df["f"].values
    xs = Re / (1 - phi)
    ys = f * phi**3 / (1 - phi)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(xs, ys, ".", ms=2, color="lightgray", label="all rows")
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    k = 0
    for s in axis_data[best_ax]["segments"]:
        idx = s["idx"]
        if s["type"] == "regime":
            ax.loglog(xs[idx], ys[idx], ".", ms=3, color=palette[k % 4],
                      label=f"discovered regime {k+1}")
            k += 1
        else:
            ax.loglog(xs[idx], ys[idx], "x", ms=3, color="k", alpha=0.4,
                      label="transition zone")
    xg = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 200)
    ax.loglog(xg, 150 / xg + 1.75, "k--", lw=1, label="Ergun master curve")
    handles, labels = ax.get_legend_handles_labels()
    seen, hh, ll = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); hh.append(h); ll.append(l)
    ax.legend(hh, ll, fontsize=8)
    ax.set_xlabel(r"$Re_p/(1-\phi)$")
    ax.set_ylabel(r"$f\,\phi^3/(1-\phi)$")
    ax.set_title("Regimes discovered by the scan — no region was specified")
    fig.tight_layout()
    fig.savefig(out_dir / "local_symmetry_regimes.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset CSV path")
    ap.add_argument("--output-dir", default="output_local_symmetry")
    ap.add_argument("--window-frac", type=float, default=0.10,
                    help="fraction of rows per window (default 0.10)")
    ap.add_argument("--n-centers", type=int, default=41,
                    help="overlapping window centers for trajectories")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare-ergun", action="store_true",
                    help="report cosines vs the known Ergun branch "
                         "exponents (synthetic-data validation only)")
    ap.add_argument("--merge-tol", type=float, default=0.10,
                    help="practical-equivalence tolerance for merging adjacent\n windows into one regime (projected exponent units)")
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

    X_plot = np.column_stack([np.log(df["Re_p"].values),
                              np.log(df["phi"].values),
                              np.log(1 - df["phi"].values)])
    axis_data, best_ax, results = run_scan(
        df, out_dir, args.window_frac, args.n_centers, args.bootstrap,
        args.seed, args.compare_ergun, not args.no_emit_regimes,
        args.merge_tol)

    for ax_i in range(3):
        plot_axis(axis_data, ax_i, out_dir, args.compare_ergun,
                  results["noise_floor"], X_plot)
    plot_regimes_on_collapse(df, axis_data, best_ax, out_dir)
    print(f"\nWrote report, JSON, and figures to {out_dir}")


if __name__ == "__main__":
    main()
