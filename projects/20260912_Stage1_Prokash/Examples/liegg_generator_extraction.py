"""
LieGG generator extraction on the keyhole and LPBF porosity examples.

Alternative to the Stage-1 closed-form path (Step 3 linear encoder +
null space of W): following Moskalev et al., "LieGG: Studying Learned
Lie Group Generators" (NeurIPS 2022), we

  1. build the SAME geometric-mean-centred Pi features each example's
     Step 3 uses, but work in natural-log coordinates u = log(Pi),
     where a scaling symmetry  Pi -> Pi * exp(eps*s)  becomes a plain
     translation  u -> u + eps*s;
  2. train one generic MLP regressor F(u) ~ y with no symmetry
     assumptions (pydimension.symmetry_discovery.liegg.train_regressor);
  3. stack the per-sample input gradients dF/du into the network
     polarization matrix E (action="translation") and SVD it:
       - the TOP right singular vectors span the directions y depends
         on — the discovered law direction(s), comparable to Step 3's W;
       - the BOTTOM (near-null) singular vectors are the Lie-algebra
         generators: scaling moves that leave the network output flat.

No symmetry-class competition, no clustering heuristics, no null-space
of a trained W — the generators come straight out of one SVD.

Grading, as in the per-example figures:
  keyhole : cos(top vector, Ke Pi-exponents); drift of Ke along each
            generator, exp(eps * c.g), max over |eps| <= 0.5.
  LPBF    : cos(top vectors, Pe_vap / Pr axes); drift of the supplied
            Pe_vap and Pr along each generator (their axis components).

Outputs per example (written to <example>/output_liegg/):
  liegg_results.json, liegg_generators.png, run printed to stdout.

Usage:
    python liegg_generator_extraction.py [--epochs 4000] [--seed 42]
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydimension.symmetry_discovery.liegg import (  # noqa: E402
    extract_liegg_generators,
    input_gradients,
    polarization_matrix,
    train_regressor,
)


def _load_module(example_dir: Path, alias: str):
    """Import an example's discover_symmetry.py under a unique alias."""
    spec = importlib.util.spec_from_file_location(
        alias, example_dir / "discover_symmetry.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    cwd = os.getcwd()
    os.chdir(example_dir)          # module-level code may use relative paths
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


# ------------------------------------------------------------------
# Feature construction (identical inputs to each example's Step 3)
# ------------------------------------------------------------------

def keyhole_features():
    ex_dir = EXAMPLES_DIR / "keyhole_symmetry"
    mod = _load_module(ex_dir, "keyhole_ds")
    out_da = ex_dir / "output_liegg" / "_da_repo"
    cwd = os.getcwd()
    os.chdir(ex_dir)
    try:
        X, y, Ke = mod.load_data(SimpleNamespace(data="dataset_keyhole.csv"))
        repo = mod.run_repo_dimensional_analysis(
            csv_path="dataset_keyhole.csv",
            input_vars=mod.VARIABLE_NAMES,
            output_var="e*",
            dim_matrix=mod.DIMENSION_MATRIX,
            output_dir=str(out_da),
        )
        pi_basis = repo["basis_vectors"]
        pi_centred = mod.compute_pi_values_centred(X, pi_basis)
    finally:
        os.chdir(cwd)

    # Ke expressed in Pi-group coordinates (c such that log Ke = c . log Pi)
    ke_coords, *_ = np.linalg.lstsq(pi_basis, mod.KNOWN_KE_EXPONENTS, rcond=None)
    return {
        "name": "keyhole",
        "out_dir": ex_dir / "output_liegg",
        "features": np.log(pi_centred),          # natural-log Pi coordinates
        "y": y,
        "feature_names": [f"Pi{i+1}" for i in range(pi_centred.shape[1])],
        "k_star": 1,
        "refs": {"Ke": ke_coords},
    }


def lpbf_features():
    ex_dir = EXAMPLES_DIR / "lpbf_porosity_symmetry"
    mod = _load_module(ex_dir, "lpbf_ds")
    out_dir = ex_dir / "output_liegg"
    da_dir = out_dir / "_da_repo"
    da_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(ex_dir)
    try:
        X, y, Pi, Pe_vap, Pr, materials = mod.load_data(
            SimpleNamespace(data="dataset_lpbf.csv"))
        enriched = da_dir / "dataset_lpbf_enriched.csv"
        mod._enrich_lpbf_csv("dataset_lpbf.csv", str(enriched))
        repo = mod.run_repo_dimensional_analysis(
            csv_path=str(enriched),
            input_vars=mod.VARIABLE_NAMES,
            output_var="Pore",
            dim_matrix=mod.DIMENSION_MATRIX,
            output_dir=str(da_dir),
        )
        pi_basis = repo["basis_vectors"]
    finally:
        os.chdir(cwd)

    # Same 8 features as the example's Step 3: 6 DA groups + Pe_vap + Pr,
    # geometric-mean centred (mean-zero in log space).
    X_pos = np.maximum(X, 1e-30)
    log_all = np.hstack([
        np.log(X_pos) @ pi_basis,
        np.log(np.maximum(np.column_stack([Pe_vap, Pr]), 1e-30)),
    ])
    log_all = log_all - log_all.mean(axis=0, keepdims=True)

    n_feat = log_all.shape[1]
    pe_axis = np.eye(n_feat)[n_feat - 2]
    pr_axis = np.eye(n_feat)[n_feat - 1]
    return {
        "name": "lpbf_porosity",
        "out_dir": out_dir,
        "features": log_all,
        "y": y,
        "feature_names": [f"Pi{i+1}" for i in range(pi_basis.shape[1])]
                         + ["Pe_vap", "Pr"],
        "k_star": 2,
        "refs": {"Pe_vap": pe_axis, "Pr": pr_axis},
    }


# ------------------------------------------------------------------
# LieGG extraction and grading
# ------------------------------------------------------------------

def run_liegg(case: dict, n_epochs: int, seed: int) -> dict:
    U = case["features"]
    y = case["y"]
    n_feat = U.shape[1]
    k_star = case["k_star"]
    n_gen = n_feat - k_star

    y_mu, y_sd = float(y.mean()), float(y.std()) + 1e-12
    y_n = (y - y_mu) / y_sd

    print(f"\n{'=' * 60}\nLieGG on {case['name']}  "
          f"({U.shape[0]} samples, {n_feat} log-Pi features, k* = {k_star})\n{'=' * 60}")

    model = train_regressor(U, y_n, hidden_dims=(64, 64),
                            n_epochs=n_epochs, seed=seed, verbose=True)

    import torch
    with torch.no_grad():
        pred = model(torch.tensor(U, dtype=torch.float32)).squeeze(1).numpy()
    r2 = 1.0 - float(np.sum((y_n - pred) ** 2)) / float(np.sum((y_n - y_n.mean()) ** 2))
    print(f"  network R^2 = {r2:.4f}")

    grads = input_gradients(model, U)
    E = polarization_matrix(grads, U, action="translation")
    extraction = extract_liegg_generators(
        E, n_inputs=n_feat, action="translation", n_generators=n_gen)
    spectrum = extraction["spectrum"]
    generators = extraction["generators"]

    # Top singular vectors = the discovered law direction(s)
    _, _, Vt = np.linalg.svd(E, full_matrices=True)
    law_dirs = [Vt[i] for i in range(k_star)]

    gap = spectrum[k_star - 1] / max(spectrum[k_star], 1e-12)
    print(f"  singular spectrum: {np.round(spectrum, 4).tolist()}")
    print(f"  spectral gap sigma_{k_star}/sigma_{k_star + 1} = {gap:.1f}x  "
          f"(law directions vs generators)")

    names = case["feature_names"]
    report = {"network_r2": r2,
              "spectrum": spectrum.tolist(),
              "spectral_gap": float(gap),
              "law_directions": [d.tolist() for d in law_dirs],
              "generators": [np.asarray(g).tolist() for g in generators],
              "feature_names": names,
              "cos_refs": {}, "drift": {}}

    for i, d in enumerate(law_dirs):
        line = ", ".join(f"{n}:{v:+.3f}" for n, v in zip(names, d))
        print(f"  law direction {i + 1}: {line}")
        for ref_name, ref in case["refs"].items():
            ref_n = ref / (np.linalg.norm(ref) + 1e-12)
            cos = float(np.dot(d, ref_n))
            report["cos_refs"][f"law{i + 1}_vs_{ref_name}"] = cos
            print(f"    cos(law {i + 1}, {ref_name}) = {cos:+.4f}")

    print(f"\n  {n_gen} generators (scaling moves Pi -> Pi * exp(eps*g)):")
    for i, g in enumerate(generators):
        line = ", ".join(f"{n}:{v:+.3f}" for n, v in zip(names, g))
        print(f"  g{i + 1}: {line}")
        for ref_name, ref in case["refs"].items():
            # drift of the reference along this orbit at eps = +-0.5:
            # log ref moves by eps * (c . g)  ->  ratio exp(0.5*|c.g|)
            drift = float(np.exp(0.5 * abs(np.dot(ref, g))) - 1.0)
            report["drift"].setdefault(ref_name, []).append(drift)
            print(f"      max |Δ{ref_name}/{ref_name}| = {drift:.3f} at |eps|=0.5")

    return report


# ------------------------------------------------------------------
# Figure
# ------------------------------------------------------------------

def make_figure(case: dict, report: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = report["feature_names"]
    spectrum = np.asarray(report["spectrum"])
    gens = np.asarray(report["generators"])
    laws = np.asarray(report["law_directions"])
    k_star = laws.shape[0]
    n_gen = gens.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle(f"LieGG generator extraction — {case['name']}",
                 fontweight="bold", fontsize=18)

    ax = axes[0]
    colors = ["#4C72B0"] * k_star + ["#DD8452"] * n_gen
    ax.bar(range(len(spectrum)), np.maximum(spectrum, 1e-12), color=colors,
           edgecolor="black", lw=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(spectrum)))
    ax.set_xticklabels([f"σ{i+1}" for i in range(len(spectrum))])
    ax.set_ylabel("singular value (log)")
    ax.set_title(f"Polarization spectrum\nblue = law directions (k*={k_star}), "
                 f"orange = generators\ngap = {report['spectral_gap']:.1f}x")

    ax = axes[1]
    M = np.vstack([laws, gens])
    labels = [f"law {i+1}" for i in range(k_star)] + \
             [f"g{i+1}" for i in range(n_gen)]
    vmax = float(np.max(np.abs(M))) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.axhline(k_star - 0.5, color="black", lw=1.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) > 0.05:
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=11,
                        color="white" if abs(M[i, j]) > 0.6 * vmax else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Law directions (top rows) and generators")

    ax = axes[2]
    eps = np.linspace(-0.5, 0.5, 41)
    cmap = plt.get_cmap("tab10")
    styles = {ref: ls for ref, ls in zip(case["refs"], ["-", ":"])}
    for ref_name, ref in case["refs"].items():
        for i, g in enumerate(gens):
            ax.plot(eps, np.exp(eps * float(np.dot(ref, g))),
                    styles[ref_name], color=cmap(i % 10), lw=1.6,
                    label=f"{ref_name}, g{i+1}")
    ax.axhline(1.0, color="grey", ls=":", lw=1.2)
    ax.set_xlabel("orbit parameter ε  (Pi → Pi · exp(ε·g))")
    ax.set_ylabel("ratio to ε=0")
    worst = max(max(v) for v in report["drift"].values())
    ax.set_title(f"Reference drift along LieGG orbits\n"
                 f"max drift = {worst:.3f} at |ε|=0.5")
    ax.legend(loc="best", ncol=2, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = case["out_dir"] / "liegg_generators.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for build in (keyhole_features, lpbf_features):
        case = build()
        case["out_dir"].mkdir(parents=True, exist_ok=True)
        report = run_liegg(case, n_epochs=args.epochs, seed=args.seed)
        with open(case["out_dir"] / "liegg_results.json", "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        fig_path = make_figure(case, report)
        print(f"\n  saved: {case['out_dir'] / 'liegg_results.json'}")
        print(f"  saved: {fig_path}")


if __name__ == "__main__":
    main()
