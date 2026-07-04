"""
Head-to-head comparison of Lie-algebra generator extraction methods.

Method A — "closed-form" (current pipeline):
    Step-3 competitive symmetry identification (single-layer encoders with
    per-class feature maps) followed by the closed-form extraction in
    symmetry_discovery/generators.py (null space of W; clustered
    antisymmetric pairs for the rotational class).

Method B — "LieGG" (Moskalev et al., NeurIPS 2022):
    Train one generic MLP regressor F on (X, y), build the network
    polarization matrix E[i, jk] = dF/dx_j(x_i) * x_i[k], and read the
    learned Lie algebra off the near-zero singular vectors of E
    (pydimension.symmetry_discovery.liegg).

Benchmark cases (n = 3 inputs, known ground-truth algebra):

    rotation      y = sin(r) + 0.1 r,  r = x1^2 + x2^2 + 2 x3^2
                  true algebra: so(D), D = diag(1, 1, 2)   (3 generators)
    boost         y = sin(r) + 0.1 r,  r = x1^2 - x2^2 + 2 x3^2
                  true algebra: so(D), D = diag(1, -1, 2)  (3 generators)
                  adversarial for the closed-form rotational path, which
                  clusters |w| and can only emit circular rotations
    scaling       y = sin(log pi) + 0.1 log pi,  pi = x1 x2^2 / x3
                  true algebra: diag(s) with [1, 2, -1] . s = 0 (2 gens)
    translation   y = sin(w . x),  true algebra: g with w . g = 0 (2 gens)
    rotation_noisy  rotation case with 20% additive output noise

Metrics per (case, method):
    detected symmetry / #generators, symmetry bias (residual of each
    extracted generator outside the true algebra, 0 = perfect), orbit
    invariance error (relative deviation of the *true* y along the orbit
    of the extracted generator, 0 = true symmetry), wall time.

Usage:
    python compare_generator_extraction.py [--quick]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
for p in (str(REPO_ROOT), str(PROJECT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_generation.scaling import generate_scaling_data
from data_generation.translational import generate_translational_data
from symmetry_discovery.identification import identify_symmetry
from symmetry_discovery.generators import extract_generators

from pydimension.symmetry_discovery.liegg import (
    extract_liegg_generators,
    input_gradients,
    orbit_invariance_error,
    polarization_matrix,
    symmetry_bias,
    train_regressor,
)

OUTPUT_DIR = PROJECT_DIR / "output_generator_comparison"
SEED = 42


# ------------------------------------------------------------------
# Ground-truth Lie algebras
# ------------------------------------------------------------------

def quadratic_stabilizer_basis(d: np.ndarray) -> list:
    """
    Basis of {A : DA + A^T D = 0} for the quadratic form r = sum_i d_i x_i^2.

    For each pair i < j the matrix  B = e_i e_j^T / d_i  -  e_j e_i^T / d_j
    satisfies the condition; the n(n-1)/2 pairs form a basis. For equal
    positive d_i, d_j this is a circular rotation; for opposite signs, a
    hyperbolic boost.
    """
    n = len(d)
    basis = []
    for i in range(n):
        for j in range(i + 1, n):
            B = np.zeros((n, n))
            B[i, j] = 1.0 / d[i]
            B[j, i] = -1.0 / d[j]
            basis.append(B)
    return basis


def diagonal_scaling_basis(exponents: np.ndarray) -> list:
    """Basis of {diag(s) : exponents . s = 0} as n x n matrices."""
    from scipy.linalg import null_space

    ns = null_space(np.atleast_2d(exponents))
    return [np.diag(ns[:, i]) for i in range(ns.shape[1])]


# ------------------------------------------------------------------
# Benchmark case definitions
# ------------------------------------------------------------------

def make_cases(n_samples: int) -> dict:
    rng_signature = {
        "rotation": np.array([1.0, 1.0, 2.0]),
        "boost": np.array([1.0, -1.0, 2.0]),
    }
    cases = {}

    for name, d in rng_signature.items():
        for noisy in ([False, True] if name == "rotation" else [False]):
            rng = np.random.default_rng(SEED)
            X = rng.standard_normal((n_samples, 3))
            r = (X**2) @ d
            y_clean = np.sin(r) + 0.1 * r
            if noisy:
                y = y_clean + 0.2 * np.std(y_clean) * rng.standard_normal(n_samples)
            else:
                y = y_clean.copy()

            def f_true(Xq, d=d):
                rq = (np.atleast_2d(Xq) ** 2) @ d
                return np.sin(rq) + 0.1 * rq

            cases[name + ("_noisy" if noisy else "")] = {
                "X": X,
                "y": y,
                "f_true": f_true,
                "n_latent": 1,
                "true_type": "rotational",
                "true_generators": quadratic_stabilizer_basis(d),
                "liegg_action": "linear",
            }

    exponents = np.array([1.0, 2.0, -1.0])
    data = generate_scaling_data(
        n_inputs=3, m_scaling_vars=1, scaling_exponents=exponents,
        n_samples=n_samples, seed=SEED,
    )

    def f_scaling(Xq, e=exponents):
        log_pi = np.log(np.abs(np.atleast_2d(Xq))) @ e
        return np.sin(log_pi) + 0.1 * log_pi

    cases["scaling"] = {
        "X": data["X"],
        "y": data["y"],
        "f_true": f_scaling,
        "n_latent": 1,
        "true_type": "scaling",
        "true_generators": diagonal_scaling_basis(exponents),
        "liegg_action": "linear",
    }

    data = generate_translational_data(
        n_inputs=3, m_orbits=1, n_samples=n_samples, seed=SEED,
    )
    w_dir = data["orbit_directions"][0]

    def f_translation(Xq, w=w_dir):
        return np.sin(np.atleast_2d(Xq) @ w)

    cases["translation"] = {
        "X": data["X"],
        "y": data["y"],
        "f_true": f_translation,
        "n_latent": 1,
        "true_type": "translational",
        "true_generators": [data["orthogonal_directions"][:, i]
                            for i in range(data["orthogonal_directions"].shape[1])],
        "liegg_action": "translation",
    }
    return cases


# ------------------------------------------------------------------
# Generator canonicalisation (common space per case for the bias metric)
# ------------------------------------------------------------------

def canonicalize(gen, source_type: str, action: str):
    """Map a generator to the case's common representation.

    Linear-action cases compare n x n matrices: closed-form scaling
    vectors s become diag(s); rotational matrices pass through.
    Translation cases compare plain n-vectors.
    """
    gen = np.asarray(gen, dtype=float)
    if action == "translation":
        return gen
    if gen.ndim == 1:  # closed-form scaling / translational vector
        return np.diag(gen) if source_type == "scaling" else gen
    return gen


# ------------------------------------------------------------------
# Method runners
# ------------------------------------------------------------------

def run_closed_form(case: dict, n_epochs: int, n_restarts: int) -> dict:
    t0 = time.time()
    result = identify_symmetry(
        case["X"], case["y"], n_latent=case["n_latent"],
        n_epochs=n_epochs, n_restarts=n_restarts, seed=SEED, device="cpu",
    )
    detected = result["symmetry_type"]
    gens = extract_generators(detected, result["encoders"][detected])
    elapsed = time.time() - t0

    action = case["liegg_action"]
    canon = [canonicalize(g, detected, action) for g in gens]
    # A generator produced under the wrong class may live in the wrong
    # space (e.g. a matrix when the case compares vectors); guard shapes.
    ref_shape = np.asarray(case["true_generators"][0]).shape
    comparable = [g for g in canon if g.shape == ref_shape]
    bias = symmetry_bias(comparable, case["true_generators"]) if comparable else np.array([])

    orbit_action = {"translational": "translation", "scaling": "linear",
                    "rotational": "linear"}[detected]
    orbit_errs = [
        orbit_invariance_error(case["f_true"], case["X"][:200],
                               canonicalize(g, detected, orbit_action),
                               action=orbit_action)
        for g in gens
    ]

    return {
        "detected_type": detected,
        "losses": {k: float(v) for k, v in result["losses"].items()},
        "n_generators": len(gens),
        "generators": [np.asarray(g).tolist() for g in gens],
        "symmetry_bias": bias.tolist(),
        "orbit_error": [float(e) for e in orbit_errs],
        "runtime_s": elapsed,
    }


def run_liegg(case: dict, n_epochs: int, n_true: int) -> dict:
    t0 = time.time()
    model = train_regressor(case["X"], case["y"], n_epochs=n_epochs, seed=SEED)
    grads = input_gradients(model, case["X"])
    E = polarization_matrix(grads, case["X"], action=case["liegg_action"])
    extraction = extract_liegg_generators(
        E, n_inputs=case["X"].shape[1], action=case["liegg_action"],
        n_generators=n_true, rel_tol=3e-2,
    )
    elapsed = time.time() - t0

    import torch
    with torch.no_grad():
        pred = model(torch.tensor(case["X"], dtype=torch.float32)).squeeze(1).numpy()
    ss_res = float(np.sum((case["y"] - pred) ** 2))
    ss_tot = float(np.sum((case["y"] - case["y"].mean()) ** 2))

    gens = extraction["generators"]
    bias = symmetry_bias(gens, case["true_generators"])
    orbit_errs = [
        orbit_invariance_error(case["f_true"], case["X"][:200], g,
                               action=case["liegg_action"])
        for g in gens
    ]

    return {
        "network_r2": 1.0 - ss_res / ss_tot,
        "n_generators": len(gens),
        "n_detected_auto": extraction["n_detected"],
        "spectrum": extraction["spectrum"].tolist(),
        "symmetry_variance": extraction["symmetry_variance"].tolist(),
        "generators": [np.asarray(g).tolist() for g in gens],
        "symmetry_bias": bias.tolist(),
        "orbit_error": [float(e) for e in orbit_errs],
        "runtime_s": elapsed,
    }


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def summarize(name: str, case: dict, cf: dict, lg: dict) -> None:
    n_true = len(case["true_generators"])
    print(f"\n=== {name} ===")
    print(f"  true symmetry     : {case['true_type']}  ({n_true} generators)")
    print(f"  --- closed-form (current) ---")
    print(f"  detected type     : {cf['detected_type']}"
          + ("  [WRONG]" if cf["detected_type"] != case["true_type"] else ""))
    print(f"  generators found  : {cf['n_generators']} / {n_true}")
    if cf["symmetry_bias"]:
        print(f"  symmetry bias     : {np.round(cf['symmetry_bias'], 4).tolist()}")
    print(f"  orbit error       : {np.round(cf['orbit_error'], 4).tolist()}")
    print(f"  runtime           : {cf['runtime_s']:.1f}s")
    print(f"  --- LieGG ---")
    print(f"  network R^2       : {lg['network_r2']:.4f}")
    print(f"  auto-detected gens: {lg['n_detected_auto']} (extracted {lg['n_generators']})")
    spec = np.asarray(lg["spectrum"])
    print(f"  spectrum tail     : {np.round(spec[-(n_true + 2):], 4).tolist()}")
    print(f"  symmetry bias     : {np.round(lg['symmetry_bias'], 4).tolist()}")
    print(f"  orbit error       : {np.round(lg['orbit_error'], 4).tolist()}")
    print(f"  runtime           : {lg['runtime_s']:.1f}s")


def make_figure(results: dict, cases: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.linalg import expm

    case_names = list(results.keys())
    fig, axes = plt.subplots(2, len(case_names), figsize=(4.2 * len(case_names), 8))

    for col, name in enumerate(case_names):
        lg = results[name]["liegg"]
        cf = results[name]["closed_form"]
        n_true = len(cases[name]["true_generators"])

        # Row 1: LieGG singular spectrum
        ax = axes[0, col]
        spec = np.asarray(lg["spectrum"])
        colors = ["#d62728" if i >= len(spec) - n_true else "#1f77b4"
                  for i in range(len(spec))]
        ax.bar(range(len(spec)), np.maximum(spec, 1e-12), color=colors)
        ax.set_yscale("log")
        ax.set_title(f"{name}\nLieGG spectrum (red = true null dim)")
        ax.set_xlabel("singular value index")
        if col == 0:
            ax.set_ylabel("singular value (log)")

        # Row 2: orbit invariance error, both methods
        ax = axes[1, col]
        labels, values, bar_colors = [], [], []
        for i, e in enumerate(cf["orbit_error"]):
            labels.append(f"CF g{i + 1}")
            values.append(max(e, 1e-8))
            bar_colors.append("#ff7f0e")
        for i, e in enumerate(lg["orbit_error"]):
            labels.append(f"LieGG g{i + 1}")
            values.append(max(e, 1e-8))
            bar_colors.append("#2ca02c")
        ax.bar(range(len(values)), values, color=bar_colors)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.axhline(0.01, color="gray", ls="--", lw=1)
        ax.set_title("orbit error (lower = truer symmetry)")
        if col == 0:
            ax.set_ylabel("relative y deviation along orbit")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # Boost-case orbit picture: what each method's generator actually does
    if "boost" in results:
        cf = results["boost"]["closed_form"]
        lg = results["boost"]["liegg"]
        fig, ax = plt.subplots(figsize=(6, 6))
        g1 = np.linspace(-3, 3, 300)
        G1, G2 = np.meshgrid(g1, g1)
        ax.contour(G1, G2, G1**2 - G2**2, levels=12, colors="gray",
                   linewidths=0.6, alpha=0.6)
        x0 = np.array([1.5, 0.5, 0.8])
        taus = np.linspace(-1.2, 1.2, 200)
        for gens, color, label in [
            (cf["generators"], "#ff7f0e", "closed-form"),
            (lg["generators"], "#2ca02c", "LieGG"),
        ]:
            best = None
            for g in gens:
                A = np.asarray(g)
                if A.ndim != 2:
                    continue
                if best is None or abs(A[0, 1]) > abs(best[0, 1]):
                    best = A
            if best is None:
                continue
            orbit = np.stack([expm(t * best) @ x0 for t in taus])
            ax.plot(orbit[:, 0], orbit[:, 1], color=color, lw=2, label=label)
        ax.plot(*x0[:2], "k*", ms=12)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xlabel("x1"); ax.set_ylabel("x2")
        ax.set_title("Boost case: level sets of $x_1^2 - x_2^2$ and extracted orbits\n"
                     "(a true symmetry orbit must follow a level set)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path.with_name("boost_orbits.png"), dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="reduced epochs/samples for a fast smoke run")
    args = parser.parse_args()

    n_samples = 500 if args.quick else 2000
    cf_epochs = 100 if args.quick else 800
    cf_restarts = 1 if args.quick else 2
    lg_epochs = 300 if args.quick else 3000

    OUTPUT_DIR.mkdir(exist_ok=True)
    cases = make_cases(n_samples)

    results = {}
    for name, case in cases.items():
        print(f"\n[{time.strftime('%H:%M:%S')}] running case '{name}' ...")
        cf = run_closed_form(case, cf_epochs, cf_restarts)
        lg = run_liegg(case, lg_epochs, n_true=len(case["true_generators"]))
        results[name] = {"closed_form": cf, "liegg": lg}
        summarize(name, case, cf, lg)

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    make_figure(results, cases, OUTPUT_DIR / "comparison.png")
    print(f"\nSaved: {OUTPUT_DIR / 'results.json'}")
    print(f"Saved: {OUTPUT_DIR / 'comparison.png'}")
    if "boost" in results:
        print(f"Saved: {OUTPUT_DIR / 'boost_orbits.png'}")


if __name__ == "__main__":
    main()
