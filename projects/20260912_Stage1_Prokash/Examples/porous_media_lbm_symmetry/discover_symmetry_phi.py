"""
Porous-media symmetry discovery using the inputs
    rho, v, d, mu, phi, (1 - phi)

These reduce to THREE dimensionless coordinates fed to the Stage1 pipeline:
    Re_p = rho * v * d / mu,   phi,   (1 - phi)

(dP_L is dropped — it is the pressure RESPONSE, collinear with the target
friction factor f, not an independent driver.) The target is f.

Because the porosity dependence of the Ergun law lives in BOTH phi and
(1 - phi), giving the model (1 - phi) explicitly lets the scaling machinery
represent the porosity powers directly instead of having to construct
(1 - phi) internally.

Usage:
    python discover_symmetry_phi.py --data dataset_combined_ergun.csv --seed 42
"""

import os
import sys
import argparse

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _c in [os.path.join(_here, "..", ".."),
           os.path.join(_here, "..", "..", "projects", "20260912_Stage1_Prokash")]:
    _c = os.path.abspath(_c)
    if os.path.isdir(os.path.join(_c, "symmetry_discovery")):
        sys.path.insert(0, _c)
        break

import torch
from preprocessing.normalize import normalize_data
from intrinsic_coordinate.discovery import discover_latent_dimension
from symmetry_discovery.identification import identify_symmetry
from symmetry_discovery.generators import extract_generators

import torch.multiprocessing as _tmp
_tmp.cpu_count = lambda: 0

COORD_NAMES = ["Re_p", "phi", "one_minus_phi"]


def load_data(path):
    import csv
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r.get("converged", "True") or "True").strip().lower() != "true":
                continue
            if (r.get("stalled", "False") or "False").strip().lower() == "true":
                continue
            try:
                rho = float(r["rho"]); v = float(r["v"]); d = float(r["d"])
                mu = float(r["mu"]); phi = float(r["phi"]); f = float(r["f"])
            except (KeyError, ValueError):
                continue
            if f <= 0 or v <= 0 or phi <= 0 or phi >= 1:
                continue
            rows.append((rho, v, d, mu, phi, f))
    arr = np.array(rows, dtype=float)
    rho, v, d, mu, phi, f = arr.T
    Re_p = rho * v * d / mu                 # from rho, v, d, mu
    one_minus_phi = 1.0 - phi
    coords = np.column_stack([Re_p, phi, one_minus_phi])   # 3 dimensionless inputs
    return coords, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_combined_ergun.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent-epochs", type=int, default=600)
    ap.add_argument("--sym-epochs", type=int, default=1500)
    ap.add_argument("--n-restarts", type=int, default=3)
    ap.add_argument("--encoder-hidden", type=int, nargs="+", default=[64, 32])
    ap.add_argument("--output-dir", default="output_porous_phi")
    args = ap.parse_args()

    path = args.data if os.path.exists(args.data) else os.path.join(_here, args.data)
    coords, f = load_data(path)
    Re_p, phi, omp = coords.T
    print(f"Loaded {len(f)} rows.  Inputs = rho, v, d, mu, phi, (1-phi)")
    print(f"  -> dimensionless coordinates {COORD_NAMES}")
    print(f"  Re_p range [{Re_p.min():.2e}, {Re_p.max():.2e}], "
          f"phi [{phi.min():.3f}, {phi.max():.3f}], f [{f.min():.2e}, {f.max():.2e}]")
    print()

    # ---- Step 1: target log10(f); features for Step 2 ----
    y_log = np.log10(np.maximum(f, 1e-30))
    log10_Re = np.log10(np.maximum(Re_p, 1e-30))
    pi_features = np.column_stack([log10_Re, phi, omp])
    fmin, fmax = pi_features.min(0), pi_features.max(0)
    rng = np.where(fmax - fmin > 1e-12, fmax - fmin, 1.0)
    pi_features_norm = (pi_features - fmin) / rng
    ymin, ymax = y_log.min(), y_log.max()
    y_norm = (y_log - ymin) / max(ymax - ymin, 1e-12)

    # ---- Step 2: latent dimension ----
    print("=" * 60); print("Step 2: intrinsic latent dimension"); print("=" * 60)
    res_latent = discover_latent_dimension(
        pi_features_norm, y_norm, max_latent=min(3, pi_features_norm.shape[1]),
        n_epochs=args.latent_epochs, n_restarts=args.n_restarts, seed=args.seed,
        encoder_hidden_dims=args.encoder_hidden, raw_input=True)
    n_latent = res_latent["optimal_n_latent"]
    print(f"  Optimal latent dimension: {n_latent}")
    for k, m in res_latent["metrics"].items():
        print(f"    k={k}: R2_train={m.get('R2_train', float('nan')):.4f}, "
              f"R2_test={m['R2']:.4f}, MSE={m['MSE']:.6f}")
    print()

    # ---- Step 3: symmetry type (Pi VALUES, geometric-mean-centred) ----
    print("=" * 60); print("Step 3: symmetry type"); print("=" * 60)
    log10_vals = np.column_stack([log10_Re,
                                  np.log10(np.maximum(phi, 1e-30)),
                                  np.log10(np.maximum(omp, 1e-30))])
    log10_vals = log10_vals - log10_vals.mean(0, keepdims=True)
    pi_centred = 10.0 ** log10_vals
    res_sym = identify_symmetry(
        pi_centred, y_norm, n_latent=n_latent,
        decoder=res_latent["best_decoder"],
        n_epochs=args.sym_epochs, n_restarts=args.n_restarts, seed=args.seed)
    print(f"  Detected symmetry: {res_sym['symmetry_type']}")
    for stype, loss in sorted(res_sym["losses"].items(), key=lambda kv: kv[1]):
        mark = " <--" if stype == res_sym["symmetry_type"] else ""
        print(f"    {stype:15s}: {loss:.6f}{mark}")
    sl = sorted(res_sym["losses"].values())
    if len(sl) >= 2 and sl[0] > 0:
        print(f"  Loss gap: {sl[1] / sl[0]:.1f}x")
    print()

    # ---- Step 4: generators ----
    print("=" * 60); print("Step 4: generators"); print("=" * 60)
    winner = res_sym["symmetry_type"]
    enc = res_sym["encoders"][winner]
    gens = extract_generators(winner, enc)
    print(f"  Symmetry: {winner}   Generators: {len(gens)}")
    for i, g in enumerate(np.atleast_2d(np.array(gens))):
        if g.ndim == 1 and g.size == len(COORD_NAMES):
            parts = [f"{COORD_NAMES[j]} x exp({g[j]:+.3f}*eps)"
                     for j in range(len(COORD_NAMES)) if abs(g[j]) > 0.05]
            print(f"  Generator {i+1}: {', '.join(parts)}")
    print()

    # ---- save artifacts + genuine trained model ----
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, "pipeline_artifacts.npz"),
             pi_centred=pi_centred, y=f, coords=coords,
             W=enc.weight_matrix, generators=np.array(gens),
             coord_names=np.array(COORD_NAMES),
             y_log_min=ymin, y_log_max=ymax)
    torch.save({"encoder": enc, "decoder": res_sym["decoders"][winner],
                "symmetry_type": winner},
               os.path.join(args.output_dir, "trained_model.pt"))
    print(f"Artifacts + trained_model.pt saved to {args.output_dir}/")
    print("=" * 60); print("COMPLETE"); print("=" * 60)
    print(f"  Inputs: rho, v, d, mu, phi, (1-phi)  ->  {COORD_NAMES}")
    print(f"  Symmetry: {winner}   Generators: {len(gens)}")


if __name__ == "__main__":
    main()
