"""Regime-aware symmetry discovery via Clustering Dimensionless Learning.

Implements the method of

    L. Zhang, Z. Xu, S. Wang, G. He,
    "Clustering dimensionless learning for multiple-physical-regime systems",
    Comput. Methods Appl. Mech. Engrg. 420 (2024) 116728.

on the combined two-regime porous-media (Ergun) dataset, and couples it to
the existing Stage-1 scaling-encoder equation extraction so that BOTH local
Ergun laws are recovered without any manual regime split.

Pipeline (paper section 2.3, adapted):

  1. Dimensional analysis.  The independent dimensionless inputs are
     Pi = (Re_p, phi); x = log Pi.  Because log Re_p spans ~12 decades while
     log phi spans <1, the logs are standardised (z-scored) exactly as the
     paper recommends in its Sec. 2.4 discussion ("normalization of
     logarithms of dimensionless quantities to make their orders of
     magnitudes comparable").
  2. Regression.  Gaussian-process regression (anisotropic RBF + white
     noise) of g(x) = ln f on the standardised x — the paper's GPR choice.
     Because f spans ~10 orders of magnitude, ln f (not f) is regressed;
     gradients of ln f w.r.t. log Pi are then local power-law exponents,
     which is the natural relevance measure for scaling laws.
  3. Gradients.  The posterior-mean gradient grad g(x) is computed
     analytically from the GPR prediction formula (paper Appendix), not by
     finite differences.
  4. Clustering.  Gradient directions are grouped with the paper's
     eigenstructure-weighted K-means (Eqs. 16-20): the similarity of a
     point to cluster Omega^I is
         Sim(grad g, Omega^I) = sum_j (lambda_j/||lambda||) (ghat . w_j)^2
     with (lambda, W) the eigenpairs of the cluster's normalised-gradient
     covariance C^I (Eq. 25, i.e. phi(x) = 1/||grad g||^2 weighting).
  5. K-selection.  The paper's two criteria are evaluated for K = 1..4:
     (a) lambda_1/lambda_n >= E per cluster, (b) cluster fraction >= eps.
  6. Per-cluster active subspace.  Eigenpairs of C^I give each regime's
     dominant dimensionless direction pi_hat = exp(x . w_1).
  7. Regime-aware equation extraction.  Each discovered cluster is fed to
     the Stage-1 scaling-encoder + Tanh-decoder extraction (same
     architecture/recipe as discover_equation_encoder_l2.py) to obtain the
     local power law.  Two adaptations: (i) the winning restart is chosen
     by the extracted power law's R^2(log f) instead of decoder test-MSE
     (see train_and_extract_best_r2 for why), and (ii) in addition to the
     full cluster, the law is extracted from the cluster 'core' — points
     with a high similarity margin (Eq. 16) — which trims the regime
     interface without using any ground truth.

Ground truth (never shown to the pipeline): the two Ergun limits
  viscous  f = 150 Re^-1 phi^-3 (1-phi)^2   for Re_p/(1-phi) << 85.7
  inertial f = 1.75        phi^-3 (1-phi)   for Re_p/(1-phi) >> 85.7

Usage:
    python discover_regimes_clustering.py \
        --data dataset_ergun_combined_widephi.csv \
        --output-dir output_regime_aware --k-detail 2 3
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import discover_equation_encoder_l2 as eq  # noqa: E402  (Stage-1 extraction)

A_VISC, B_INER = 150.0, 1.75
RE_CROSS = A_VISC / B_INER

TRUTH = {
    "viscous": {"exps": [-1.0, -3.0, +2.0], "C": 150.0,
                "law": "f = 150 * Re_p^-1 * phi^-3 * (1-phi)^+2"},
    "inertial": {"exps": [0.0, -3.0, +1.0], "C": 1.75,
                 "law": "f = 1.75 * Re_p^0 * phi^-3 * (1-phi)^+1"},
}


def train_and_extract_best_r2(Re, phi, f, n_epochs, n_restarts, split_seed,
                              actual_exps):
    """Multi-restart scaling-encoder training (same architecture and recipe
    as discover_equation_encoder_l2.train_and_extract), but the winning
    restart is the one whose extracted 1-D power law has the highest
    R^2(log f) — not the lowest decoder test-MSE.

    Rationale: on the phi/(1-phi) data manifold there is a one-parameter
    family of (w, decoder) pairs with near-identical decoder MSE (see the
    README identifiability caveat), so test-MSE cannot distinguish them.
    The final report claims f is a monomial in the Pi's; among equally
    well-fitting encoders, the direction most consistent with a *global
    power law* is the one to report.  R^2(log f) is sign-invariant and uses
    no ground truth.
    """
    import torch
    import torch.nn as nn

    one = 1.0 - phi
    X_log = np.column_stack([np.log10(Re), np.log10(phi), np.log10(one)])
    X_log = X_log - X_log.mean(axis=0, keepdims=True)
    X_lin = 10.0 ** X_log
    y = np.log10(f)
    y_norm = ((y - y.min()) / (y.max() - y.min())).reshape(-1, 1)

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(y_norm))
    ntr = int(0.8 * len(y_norm))
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = torch.tensor(X_lin[tr], dtype=torch.float32)
    ytr = torch.tensor(y_norm[tr], dtype=torch.float32)
    Xte = torch.tensor(X_lin[te], dtype=torch.float32)
    yte = torch.tensor(y_norm[te], dtype=torch.float32)

    best, best_disc = {"R2_logf": -np.inf}, None
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        enc, dec = eq.ScalingEncoder(3, 1), eq.Decoder(1, 64)
        opt = torch.optim.Adam(
            list(enc.parameters()) + list(dec.parameters()),
            lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        for _ in range(n_epochs):
            opt.zero_grad()
            loss_fn(dec(enc(Xtr)), ytr).backward()
            opt.step()
        with torch.no_grad():
            te_loss = loss_fn(dec(enc(Xte)), yte).item()
            tr_loss = loss_fn(dec(enc(Xtr)), ytr).item()
        w = enc.W.weight.detach().cpu().numpy().reshape(-1)
        disc = eq.discovered_equation(w, Re, phi, f, actual_exps)
        print(f"    restart seed={seed}  test MSE={te_loss:.2e}  "
              f"power-law R2(log f)={disc['R2_logf']:.4f}")
        if disc["R2_logf"] > best["R2_logf"]:
            best = {
                "R2_logf": disc["R2_logf"], "loss_te": te_loss,
                "loss_tr": tr_loss, "w": w, "seed": seed,
                "enc_state": {k: v.clone() for k, v in enc.state_dict().items()},
                "dec_state": {k: v.clone() for k, v in dec.state_dict().items()},
            }
            best_disc = disc
    return best, best_disc


class Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ----------------------------------------------------------------------
# Steps 2-3: GPR regression + analytic posterior-mean gradients
# ----------------------------------------------------------------------

def fit_gpr(X, y, seed=0):
    kernel = (ConstantKernel(1.0, (1e-3, 1e4))
              * RBF(length_scale=np.ones(X.shape[1]),
                    length_scale_bounds=(1e-2, 1e3))
              + WhiteKernel(noise_level=1e-2,
                            noise_level_bounds=(1e-8, 1e1)))
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                   n_restarts_optimizer=3,
                                   random_state=seed)
    gpr.fit(X, y)
    return gpr


def gpr_gradients(gpr, X):
    """Analytic gradient of the GPR posterior mean at each row of X
    (paper Appendix: differentiate the prediction formula, no finite
    differences)."""
    prod = gpr.kernel_.k1            # ConstantKernel * RBF
    sigma2 = prod.k1.constant_value
    ls = np.atleast_1d(prod.k2.length_scale).astype(float)
    Xtr = gpr.X_train_
    alpha = gpr.alpha_.reshape(-1)
    y_std = getattr(gpr, "_y_train_std", 1.0)

    diff = (Xtr[None, :, :] - X[:, None, :])            # (m, n, d)
    d2 = ((diff / ls) ** 2).sum(axis=2)                 # (m, n)
    K = sigma2 * np.exp(-0.5 * d2)                      # (m, n)
    return y_std * np.einsum("mn,n,mnd->md", K, alpha, diff / ls**2)


# ----------------------------------------------------------------------
# Step 4: eigenstructure-weighted gradient clustering (paper Eqs. 16-20)
# ----------------------------------------------------------------------

def cluster_eigpairs(G_hat):
    """C^I of Eq. (25) for one cluster of unit gradients, eigenpairs in
    descending order."""
    C = G_hat.T @ G_hat / max(len(G_hat), 1)
    lam, W = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return lam[order], W[:, order]


def similarity(G_hat, lam, W):
    """Sim(grad g, Omega^I) of Eq. (16) for every point (rows of G_hat)."""
    proj2 = (G_hat @ W) ** 2
    return proj2 @ (lam / (np.linalg.norm(lam) + 1e-300))


def cluster_gradients(G_hat, K, n_restarts=20, n_iter=100, seed=0):
    """Paper's clustering algorithm (Sec. 2.2): iterate similarity matching
    and eigenpair updates from random initial partitions; keep the restart
    with the largest total similarity S (Eq. 19)."""
    rng = np.random.default_rng(seed)
    N = len(G_hat)
    best = {"S": -np.inf}
    for _ in range(n_restarts):
        labels = rng.integers(0, K, size=N)
        for _ in range(n_iter):
            eig = []
            for I in range(K):
                pts = G_hat[labels == I]
                if len(pts) < 2:      # re-seed empty/degenerate cluster
                    pts = G_hat[rng.choice(N, size=max(2, N // (4 * K)),
                                           replace=False)]
                eig.append(cluster_eigpairs(pts))
            sims = np.column_stack(
                [similarity(G_hat, lam, W) for lam, W in eig])
            new_labels = sims.argmax(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        S = sims[np.arange(N), labels].sum()
        if S > best["S"]:
            best = {"S": S, "labels": labels.copy(), "eig": eig}
    return best


def order_clusters_by_Re(labels, logRe, K):
    """Relabel clusters by ascending mean log Re_p so cluster 0 is the
    lowest-Re (viscous side) — purely cosmetic, for stable reports."""
    means = [logRe[labels == I].mean() if (labels == I).any() else np.inf
             for I in range(K)]
    order = np.argsort(means)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[v] for v in labels])


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def regime_purity(labels, true_regime, K):
    """Map each cluster to its majority true regime; return per-cluster
    majority label + fraction, and overall agreement."""
    info, correct = [], 0
    for I in range(K):
        m = labels == I
        if not m.any():
            info.append(("empty", 0.0, 0))
            continue
        counts = true_regime[m].value_counts()
        maj = counts.idxmax()
        info.append((maj, counts.max() / m.sum(), int(m.sum())))
        correct += int(counts.max())
    return info, correct / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_ergun_combined_widephi.csv")
    ap.add_argument("--output-dir", default="output_regime_aware")
    ap.add_argument("--k-max", type=int, default=4)
    ap.add_argument("--k-detail", type=int, nargs="+", default=[2, 3],
                    help="K values to analyse in detail (per-cluster active "
                         "subspace + equation extraction)")
    ap.add_argument("--E", type=float, default=50.0,
                    help="eigenvalue-ratio threshold of paper criterion (a)")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="min cluster fraction of paper criterion (b)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eq-epochs", type=int, default=1500)
    ap.add_argument("--eq-restarts", type=int, default=8)
    ap.add_argument("--eq-split-seed", type=int, default=42)
    ap.add_argument("--core-frac", type=float, default=0.25,
                    help="Additionally extract each cluster's law from its "
                         "'core': drop this fraction of points with the "
                         "lowest similarity margin (own-cluster Sim minus "
                         "best other-cluster Sim, Eq. 16). The margin is "
                         "smallest at regime interfaces, so this trims the "
                         "transition zone without using any ground truth. "
                         "0 disables.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = _here / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out_dir / "run.log")

    csv_path = args.data if os.path.isabs(args.data) else _here / args.data
    df = pd.read_csv(csv_path)
    df = df[(df["f"] > 0) & (df["Re_p"] > 0)
            & (df["phi"] > 0) & (df["phi"] < 1)].reset_index(drop=True)

    print("=" * 72)
    print("Clustering dimensionless learning (Zhang et al., CMAME 2024)")
    print("on the combined two-regime Ergun dataset")
    print("=" * 72)
    print(f"Dataset : {os.path.basename(str(csv_path))}   rows = {len(df)}")
    print(f"Re_p    : [{df.Re_p.min():.2e}, {df.Re_p.max():.2e}]")
    print(f"phi     : [{df.phi.min():.3f}, {df.phi.max():.3f}]")
    print(f"True crossover (reference only): Re_p/(1-phi) = {RE_CROSS:.1f}")

    # ---- Step 1: log Pi coordinates, standardised --------------------
    x_raw = np.column_stack([np.log(df.Re_p.values),
                             np.log(df.phi.values)])
    x_mean, x_std = x_raw.mean(axis=0), x_raw.std(axis=0)
    X = (x_raw - x_mean) / x_std
    y = np.log(df.f.values)
    print("\n[Step 1] x = log(Re_p, phi), z-scored per component")
    print(f"  std(log Re_p) = {x_std[0]:.3f}, std(log phi) = {x_std[1]:.3f}")

    # ---- Step 2: GPR regression --------------------------------------
    print("\n[Step 2] GPR of g = ln f on standardised log-Pi coordinates")
    gpr = fit_gpr(X, y, seed=args.seed)
    print(f"  fitted kernel: {gpr.kernel_}")
    r2 = gpr.score(X, y)
    print(f"  R^2(train) = {r2:.6f}")

    # ---- Step 3: analytic gradients -----------------------------------
    G = gpr_gradients(gpr, X)                    # d ln f / d x~  (standardised)
    G_raw = G / x_std                            # d ln f / d ln Pi (raw logs)
    G_norm = np.linalg.norm(G, axis=1, keepdims=True)
    G_hat = G / np.clip(G_norm, 1e-12, None)     # phi(x) = 1/||grad g||^2
    print("\n[Step 3] analytic posterior-mean gradients")
    print("  raw local exponents (d ln f / d ln Re_p): "
          f"min {G_raw[:, 0].min():+.3f}  max {G_raw[:, 0].max():+.3f}")
    print("  raw local exponents (d ln f / d ln phi) : "
          f"min {G_raw[:, 1].min():+.3f}  max {G_raw[:, 1].max():+.3f}")

    # ---- Steps 4-5: clustering + K-selection criteria -----------------
    print("\n[Steps 4-5] gradient clustering, K = 1 .."
          f" {args.k_max}   (E = {args.E:g}, eps = {args.eps:g})")
    logRe = x_raw[:, 0]
    results = {}
    print(f"\n  {'K':>2} {'cluster':>8} {'size':>6} {'frac':>7} "
          f"{'lam1':>8} {'lam2':>8} {'lam1/lam2':>10} "
          f"{'crit(a)':>8} {'crit(b)':>8}")
    for K in range(1, args.k_max + 1):
        res = cluster_gradients(G_hat, K, seed=args.seed)
        res["labels"] = order_clusters_by_Re(res["labels"], logRe, K)
        # recompute eigenpairs on final (re-ordered) labels
        res["eig"] = [cluster_eigpairs(G_hat[res["labels"] == I])
                      for I in range(K)]
        results[K] = res
        ok_a = ok_b = True
        for I in range(K):
            lam, _ = res["eig"][I]
            n_I = int((res["labels"] == I).sum())
            frac = n_I / len(df)
            ratio = lam[0] / max(lam[-1], 1e-300)
            ca, cb = ratio >= args.E, frac >= args.eps
            ok_a &= ca
            ok_b &= cb
            print(f"  {K:>2} {I:>8} {n_I:>6} {frac:>7.1%} "
                  f"{lam[0]:>8.4f} {lam[-1]:>8.2e} {ratio:>10.1f} "
                  f"{'yes' if ca else 'NO':>8} {'yes' if cb else 'NO':>8}")
        res["ok"] = (ok_a, ok_b)
        print(f"     -> K = {K}: criterion (a) "
              f"{'satisfied' if ok_a else 'violated'}, "
              f"criterion (b) {'satisfied' if ok_b else 'violated'}")

    # ---- Steps 6-7 for the requested detail K values ------------------
    for K in args.k_detail:
        res = results[K]
        labels = res["labels"]
        print("\n" + "=" * 72)
        print(f"Detailed regime analysis, K = {K}")
        print("=" * 72)

        purity, agree = regime_purity(labels, df["regime_true"], K)
        for I in range(K):
            m = labels == I
            maj, frac, n_I = purity[I]
            Re_r = df.Re_p[m] / (1 - df.phi[m])
            print(f"\n  cluster {I}: n = {n_I}  "
                  f"Re_p/(1-phi) in [{Re_r.min():.2e}, {Re_r.max():.2e}]")
            print(f"    majority true regime: {maj} ({frac:.1%} of cluster)")

            lam, W = res["eig"][I]
            print(f"    active-subspace eigenvalues: "
                  f"lam = [{lam[0]:.4f}, {lam[1]:.4f}]  "
                  f"ratio = {lam[0]/max(lam[1], 1e-300):.1f}")
            # dominant direction back in raw log-Pi coordinates
            w1 = W[:, 0]
            g_bar = G_hat[m].mean(axis=0)
            if np.dot(w1, g_bar) < 0:
                w1 = -w1
            e_raw = w1 / x_std
            e_raw = e_raw / np.max(np.abs(e_raw))
            print(f"    dominant direction w1 (standardised): "
                  f"[{w1[0]:+.4f}, {w1[1]:+.4f}]")
            print(f"    pi_hat ~ Re_p^{e_raw[0]:+.3f} * phi^{e_raw[1]:+.3f}"
                  f"   (raw-log exponents, max-normalised)")
            gr = G_raw[m].mean(axis=0)
            print(f"    mean local exponents  d ln f/d ln Re_p = {gr[0]:+.3f}"
                  f"   d ln f/d ln phi = {gr[1]:+.3f}")

        print(f"\n  agreement with true regime split "
              f"(majority mapping): {agree:.1%}")

        # save assignments
        out_csv = out_dir / f"cluster_assignments_K{K}.csv"
        pd.DataFrame({
            "Re_p": df.Re_p, "phi": df.phi, "f": df.f,
            "regime_true": df.regime_true, "cluster": labels,
        }).to_csv(out_csv, index=False)
        print(f"  wrote {out_csv.name}")

        # similarity margin: how unambiguously each point belongs to its
        # cluster (Eq. 16); smallest at the regime interface
        sims = np.column_stack(
            [similarity(G_hat, lam, W) for lam, W in res["eig"]])
        own = sims[np.arange(len(df)), labels]
        other = np.where(
            np.eye(K, dtype=bool)[labels], -np.inf, sims).max(axis=1) \
            if K > 1 else np.zeros(len(df))
        margin = own - other

        # ---- Step 7: per-cluster Stage-1 equation extraction ----------
        print(f"\n[Step 7] regime-aware equation extraction (K = {K})")
        for I in range(K):
            m = labels == I
            maj = purity[I][0]
            truth = TRUTH[maj]

            subsets = [("full", m)]
            if args.core_frac > 0 and K > 1:
                thr = np.quantile(margin[m], args.core_frac)
                core = m & (margin >= thr)
                subsets.append(("core", core))

            for tag, sel in subsets:
                Re_c, phi_c, f_c = (df.Re_p[sel].values, df.phi[sel].values,
                                    df.f[sel].values)
                print(f"\n  --- cluster {I} [{tag}] ({int(sel.sum())} pts, "
                      f"expected: {maj} limit) ---")
                if tag == "core":
                    print(f"    (dropped the {args.core_frac:.0%} of the "
                          f"cluster with the lowest similarity margin)")
                best, disc = train_and_extract_best_r2(
                    Re_c, phi_c, f_c, args.eq_epochs, args.eq_restarts,
                    args.eq_split_seed, truth["exps"])
                suffix = "" if tag == "full" else "_core"
                eq.write_report(
                    out_dir / f"discovered_equation_K{K}_cluster{I}{suffix}.txt",
                    dataset_name=(f"{os.path.basename(str(csv_path))} "
                                  f"[K={K}, cluster {I}, {tag}]"),
                    actual_str=truth["law"], actual_exps=truth["exps"],
                    actual_C=truth["C"], best=best, disc=disc,
                    n_epochs=args.eq_epochs, n_restarts=args.eq_restarts)
                e = disc["exps"]
                print(f"    discovered: f = {disc['C']:.4g}"
                      f" * Re_p^{e[0]:+.3f} * phi^{e[1]:+.3f}"
                      f" * (1-phi)^{e[2]:+.3f}")
                print(f"    actual    : {truth['law']}")
                print(f"    cos vs Ergun (raw 3-D) = {disc['cos_raw']:+.4f}   "
                      f"(manifold) = {disc['cos_manifold']:+.4f}   "
                      f"R2(f) = {disc['R2_f']:.4f}")

        make_figures(df, labels, res, G, G_hat, K, out_dir)

    print(f"\nAll outputs in {out_dir}")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def make_figures(df, labels, res, G, G_hat, K, out_dir):
    colors = plt.cm.tab10(np.arange(K))
    Re_r = df.Re_p / (1 - df.phi)
    f_r = df.f * df.phi**3 / (1 - df.phi)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    xx = np.logspace(np.log10(Re_r.min()), np.log10(Re_r.max()), 400)
    for ax, lab, title in (
        (axes[0], labels, f"discovered clusters (K = {K})"),
        (axes[1], (df.regime_true == "inertial").astype(int).values,
         "true regimes (reference)"),
    ):
        ax.plot(xx, A_VISC / xx + B_INER, "k--", lw=1,
                label="Ergun master curve")
        for I in np.unique(lab):
            m = lab == I
            name = (f"cluster {I}" if title.startswith("discovered")
                    else ["viscous", "inertial"][I])
            ax.scatter(Re_r[m], f_r[m], s=8, alpha=0.6,
                       color=colors[I % 10], label=name)
        ax.axvline(RE_CROSS, color="gray", lw=1, ls=":",
                   label=f"crossover {RE_CROSS:.0f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$Re_p/(1-\varphi)$")
        ax.set_ylabel(r"$f\,\varphi^3/(1-\varphi)$")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Regime identification by clustering dimensionless learning")
    fig.tight_layout()
    fig.savefig(out_dir / f"regime_clusters_master_curve_K{K}.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    for I in range(K):
        m = labels == I
        ax.scatter(G[m, 0], G[m, 1], s=8, alpha=0.6, color=colors[I % 10],
                   label=f"cluster {I}")
        lam, W = res["eig"][I]
        g_bar = G_hat[m].mean(axis=0)
        w1 = W[:, 0] if np.dot(W[:, 0], g_bar) >= 0 else -W[:, 0]
        scale = 0.9 * np.linalg.norm(G[m], axis=1).mean()
        ax.annotate("", xy=w1 * scale, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2,
                                    color=colors[I % 10]))
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"$\partial \ln f/\partial \tilde{x}_{Re}$")
    ax.set_ylabel(r"$\partial \ln f/\partial \tilde{x}_{\varphi}$")
    ax.set_title("GPR gradients in standardised log-Pi space\n"
                 "(arrows: per-cluster dominant eigenvector)")
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.8 / K
    for I in range(K):
        lam, _ = res["eig"][I]
        ax.bar(np.arange(len(lam)) + I * width, lam, width,
               color=colors[I % 10], label=f"cluster {I}")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(G.shape[1]) + 0.4 - width / 2)
    ax.set_xticklabels([f"$\\lambda_{j+1}$" for j in range(G.shape[1])])
    ax.set_title("Active-subspace eigenvalues per cluster (Eq. 25)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"regime_clusters_gradients_K{K}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
