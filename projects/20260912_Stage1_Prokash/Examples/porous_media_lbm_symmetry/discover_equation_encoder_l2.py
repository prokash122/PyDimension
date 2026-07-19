"""Extract the discovered equation from the winning scaling encoder's
L2-normed weight vector — no L1, no OLS-on-log-features, nothing extra.

The recipe (from README):

  1. Train a scaling encoder z = w . log|pi| jointly with a Tanh MLP decoder
     on log10(f), using plain L2 (weight_decay = 1e-4).  Multi-restart, keep
     best test MSE.
  2. L2-normalise w and sign-align so w . [log Re, log phi, log(1-phi)]
     correlates positively with log f.
  3. Fit a scalar magnitude alpha and prefactor C by 1-D OLS:
        log f = alpha * (w . x) + log C
     Discovered law: f = C * Re_p^(alpha*w1) * phi^(alpha*w2) * (1-phi)^(alpha*w3).

Writes discovered_equation.txt + trained_encoder.pt to the given output dir.
"""
import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_dataset(csv_path):
    Re, phi, f = [], [], []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if (r.get("converged", "False") or "False").lower() != "true":
                continue
            if (r.get("stalled", "False") or "False").lower() == "true":
                continue
            try:
                fv = float(r["f"]); Rv = float(r["Re_p"]); pv = float(r["phi"])
                if fv > 0 and Rv > 0 and 0 < pv < 1:
                    Re.append(Rv); phi.append(pv); f.append(fv)
            except (KeyError, ValueError):
                continue
    return np.array(Re), np.array(phi), np.array(f)


class ScalingEncoder(nn.Module):
    def __init__(self, n_in=3, n_lat=1):
        super().__init__()
        self.W = nn.Linear(n_in, n_lat, bias=False)

    def forward(self, x):
        return self.W(torch.log(x.abs().clamp(min=1e-6)))


class Decoder(nn.Module):
    def __init__(self, n_lat=1, h=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_lat, h), nn.Tanh(),
            nn.Linear(h, h),     nn.Tanh(),
            nn.Linear(h, 1),
        )

    def forward(self, z):
        return self.net(z)


def train_and_extract(Re, phi, f, n_epochs, n_restarts, split_seed):
    """Train encoder+decoder end-to-end; return the winning restart's info."""
    one_minus_phi = 1 - phi
    log10_Re = np.log10(Re)
    log10_phi = np.log10(phi)
    log10_omp = np.log10(one_minus_phi)
    X_log = np.column_stack([log10_Re, log10_phi, log10_omp])
    X_log = X_log - X_log.mean(axis=0, keepdims=True)
    X_lin = 10.0 ** X_log

    y = np.log10(f)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)
    y_norm = y_norm.reshape(-1, 1)

    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(y_norm))
    ntr = int(0.8 * len(y_norm))
    tr, te = idx[:ntr], idx[ntr:]

    Xtr = torch.tensor(X_lin[tr], dtype=torch.float32)
    ytr = torch.tensor(y_norm[tr], dtype=torch.float32)
    Xte = torch.tensor(X_lin[te], dtype=torch.float32)
    yte = torch.tensor(y_norm[te], dtype=torch.float32)

    best = {"loss_te": float("inf")}
    for seed in range(n_restarts):
        torch.manual_seed(seed)
        enc = ScalingEncoder(3, 1)
        dec = Decoder(1, 64)
        opt = torch.optim.Adam(
            list(enc.parameters()) + list(dec.parameters()),
            lr=1e-3, weight_decay=1e-4,
        )
        loss_fn = nn.MSELoss()
        for _ in range(n_epochs):
            opt.zero_grad()
            loss_fn(dec(enc(Xtr)), ytr).backward()
            opt.step()
        with torch.no_grad():
            te_loss = loss_fn(dec(enc(Xte)), yte).item()
            tr_loss = loss_fn(dec(enc(Xtr)), ytr).item()
        print(f"  restart seed={seed}  train MSE={tr_loss:.6f}  test MSE={te_loss:.6f}")
        if te_loss < best["loss_te"]:
            w = enc.W.weight.detach().cpu().numpy().reshape(-1)
            best = {
                "loss_te": te_loss, "loss_tr": tr_loss,
                "w": w, "seed": seed,
                "enc_state": {k: v.clone() for k, v in enc.state_dict().items()},
                "dec_state": {k: v.clone() for k, v in dec.state_dict().items()},
            }
    return best


def discovered_equation(w_raw, Re, phi, f, actual_exps):
    """Return dict describing the discovered law extracted from w_raw."""
    w_norm = w_raw / (np.linalg.norm(w_raw) + 1e-12)
    truth_n = np.asarray(actual_exps, dtype=float)
    truth_n = truth_n / np.linalg.norm(truth_n)
    if float(np.dot(w_norm, truth_n)) < 0:
        w_norm = -w_norm

    cos_raw = float(np.dot(w_norm, truth_n))
    phi_bar = float(phi.mean())
    slope = -phi_bar / (1 - phi_bar)
    proj = lambda v: np.array([v[0], v[1] + slope * v[2]])
    we, tr = proj(w_norm), proj(truth_n)
    cos_manifold = float(np.dot(we, tr) / (np.linalg.norm(we) * np.linalg.norm(tr)))

    logRe = np.log(Re)
    logphi = np.log(phi)
    log1mphi = np.log(1 - phi)
    z = w_norm[0] * logRe + w_norm[1] * logphi + w_norm[2] * log1mphi
    logf = np.log(f)
    A = np.column_stack([z, np.ones_like(z)])
    alpha, logC = np.linalg.lstsq(A, logf, rcond=None)[0]
    exps = alpha * w_norm
    C = math.exp(logC)
    pred = C * Re ** exps[0] * phi ** exps[1] * (1 - phi) ** exps[2]
    R2_f = 1 - float(np.sum((f - pred) ** 2) / np.sum((f - f.mean()) ** 2))
    R2_logf = 1 - float(
        np.sum((logf - (logC + alpha * z)) ** 2)
        / np.sum((logf - logf.mean()) ** 2)
    )

    return {
        "w_raw": w_raw, "w_norm": w_norm, "phi_bar": phi_bar,
        "cos_raw": cos_raw, "cos_manifold": cos_manifold,
        "alpha": float(alpha), "logC": float(logC), "C": C,
        "exps": exps, "R2_f": R2_f, "R2_logf": R2_logf,
    }


def write_report(path, dataset_name, actual_str, actual_exps, actual_C,
                 best, disc, n_epochs, n_restarts):
    def fmt(x, k=4): return f"{x:+.{k}f}"
    def fmt_arr(a, k=4): return "[" + ", ".join(fmt(v, k) for v in a) + "]"

    lines = []
    lines.append(f"Dataset                 : {dataset_name}")
    lines.append(f"Architecture            : scaling encoder (z = w . log|pi|)"
                 f"  +  Tanh MLP decoder [1 -> 64 -> 64 -> 1]")
    lines.append(f"Training                : Adam, lr=1e-3, weight_decay=1e-4, "
                 f"MSE on min-max(log10 f),  {n_epochs} epochs, {n_restarts} restarts (best kept)")
    lines.append(f"Best restart seed       : {best['seed']}  "
                 f"(train MSE = {best['loss_tr']:.6f}, test MSE = {best['loss_te']:.6f})")
    lines.append("")
    lines.append(f"Encoder raw weight w     : {fmt_arr(disc['w_raw'])}")
    lines.append(f"Encoder L2-normed w      : {fmt_arr(disc['w_norm'])}"
                 f"      on [log Re_p, log phi, log(1-phi)]")
    lines.append(f"phi_bar                  : {disc['phi_bar']:.4f}")
    lines.append(f"cos vs Ergun (raw 3-D)   : {disc['cos_raw']:+.4f}")
    lines.append(f"cos vs Ergun (manifold)  : {disc['cos_manifold']:+.4f}")
    lines.append("")
    lines.append(f"Actual (Ergun) : {actual_str}")
    lines.append(f"  exponents          = {actual_exps}")
    lines.append(f"  prefactor C_actual = {actual_C}")
    lines.append("")
    lines.append("1-D OLS on log f = alpha*(w . x) + log C:")
    lines.append(f"  alpha  = {disc['alpha']:+.4f}   log C = {disc['logC']:+.4f}   C = {disc['C']:.4g}")
    lines.append(f"  exponents (alpha * w_norm) = {fmt_arr(disc['exps'])}")
    lines.append("")
    lines.append(f"Discovered law (encoder L2 only):")
    lines.append(
        f"    f = {disc['C']:.4g} * Re_p^{disc['exps'][0]:+.3f}"
        f" * phi^{disc['exps'][1]:+.3f}"
        f" * (1-phi)^{disc['exps'][2]:+.3f}"
    )
    lines.append(f"  R2(log f) = {disc['R2_logf']:.4f}")
    lines.append(f"  R2(f)     = {disc['R2_f']:.4f}")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset CSV path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--region", choices=["viscous", "inertial"], required=True)
    ap.add_argument("--n-epochs", type=int, default=1500)
    ap.add_argument("--n-restarts", type=int, default=3)
    ap.add_argument("--split-seed", type=int, default=0)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    csv_path = args.data if os.path.exists(args.data) else str(here / args.data)
    Re, phi, f = load_dataset(csv_path)
    print(f"Loaded {len(f)} rows from {csv_path}")
    print(f"  Re range [{Re.min():.2e}, {Re.max():.2e}]  "
          f"phi range [{phi.min():.3f}, {phi.max():.3f}]  "
          f"f range [{f.min():.3g}, {f.max():.3g}]")

    if args.region == "viscous":
        actual_exps = [-1.0, -3.0, +2.0]
        actual_C = 150.0
        actual_str = "f = 150 * Re_p^-1 * phi^-3 * (1-phi)^+2"
    else:
        actual_exps = [0.0, -3.0, +1.0]
        actual_C = 1.75
        actual_str = "f = 1.75 * Re_p^0 * phi^-3 * (1-phi)^+1"

    best = train_and_extract(Re, phi, f, args.n_epochs, args.n_restarts, args.split_seed)
    disc = discovered_equation(best["w"], Re, phi, f, actual_exps)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = here / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_report(
        out_dir / "discovered_equation.txt",
        dataset_name=os.path.basename(csv_path),
        actual_str=actual_str, actual_exps=actual_exps, actual_C=actual_C,
        best=best, disc=disc,
        n_epochs=args.n_epochs, n_restarts=args.n_restarts,
    )
    torch.save(
        {"encoder": best["enc_state"], "decoder": best["dec_state"],
         "w_raw": best["w"], "w_norm": disc["w_norm"],
         "alpha": disc["alpha"], "logC": disc["logC"]},
        out_dir / "trained_encoder_l2.pt",
    )
    print(f"\nWrote {out_dir/'discovered_equation.txt'}")
    print(f"Wrote {out_dir/'trained_encoder_l2.pt'}")

    print()
    print("Discovered law: "
          f"f = {disc['C']:.4g} * Re_p^{disc['exps'][0]:+.3f}"
          f" * phi^{disc['exps'][1]:+.3f}"
          f" * (1-phi)^{disc['exps'][2]:+.3f}")
    print(f"R2(f) = {disc['R2_f']:.4f}")


if __name__ == "__main__":
    main()
