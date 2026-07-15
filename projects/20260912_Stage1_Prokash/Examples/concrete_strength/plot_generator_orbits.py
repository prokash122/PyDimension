"""
Simple demonstration that the discovered generators preserve the model
output: take real mixes, slide them along each generator direction,
feed every point to the trained model, and plot the prediction.

The trained translational encoder computes z = W*pi, and each generator
g satisfies W*g = 0. Therefore

    model(pi + eps*g) = f(W*pi + eps*W*g) = f(W*pi)   for every eps:

the prediction cannot change along a generator. As contrast, the same
sweep along the most strength-relevant direction (top right-singular
vector of W) changes the prediction strongly.

The decoder head f (latent z -> strength residual) is refit on the
frozen latent coordinates of the run of record stored in
pipeline_artifacts.npz; the invariance itself depends only on W.

Usage
-----
    python plot_generator_orbits.py     # after discover_symmetry_dimensionless.py
"""

import os
import argparse

import numpy as np
import torch
import torch.nn as nn

try:
    import matplotlib
    matplotlib.use("Agg")
except (AttributeError, ImportError):
    pass
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size":       12,
    "axes.titlesize":  13,
    "axes.labelsize":  13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

_here = os.path.dirname(os.path.abspath(__file__))


def fit_head(z: np.ndarray, y: np.ndarray, seed: int = 0) -> nn.Module:
    """Refit the small decoder head f(z) -> y on frozen latents."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = nn.Sequential(
        nn.Linear(z.shape[1], 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )
    zt = torch.tensor(z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(3000):
        opt.zero_grad()
        loss = loss_fn(model(zt), yt)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(zt).numpy().ravel()
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    print(f"Decoder head refit on frozen z: R2 = {1 - ss_res/ss_tot:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Orbit plots: model output along generator directions")
    parser.add_argument("--artifacts",
                        default="output_concrete_dimensionless/pipeline_artifacts.npz")
    parser.add_argument("--output-dir", default="output_concrete_dimensionless")
    parser.add_argument("--eps-max", type=float, default=1.5)
    args = parser.parse_args()

    path = args.artifacts
    if not os.path.exists(path):
        path = os.path.join(_here, args.artifacts)
    data = np.load(path, allow_pickle=True)
    X = data["X_norm"]          # standardized pi features (n, 7)
    y = data["y"]               # measured sigma / sigma_ideal
    W = data["W"]               # (4, 7)
    gens = data["generators"]   # (3, 7)

    print(f"Loaded {X.shape[0]} mixes; k = {W.shape[0]}, "
          f"{gens.shape[0]} generators")
    print(f"max |W @ g| over generators: "
          f"{np.abs(W @ gens.T).max():.2e}  (0 = perfect symmetry)")

    head = fit_head(X @ W.T, y)

    def predict(Xq: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.tensor(Xq @ W.T, dtype=torch.float32)
            return head(z).numpy().ravel()

    # Directions to sweep: the three generators + the strength direction.
    g_unit = gens / np.linalg.norm(gens, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(W)
    v1 = Vt[0]                                  # most strength-relevant
    directions = [(f"generator g{i+1}", g) for i, g in enumerate(g_unit)]
    directions.append(("strength direction (control)", v1))

    # Three representative real mixes: weak / typical / strong residual.
    order = np.argsort(y)
    picks = [order[int(len(order) * q)] for q in (0.10, 0.50, 0.90)]
    labels = ["weak mix", "typical mix", "strong mix"]
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    eps = np.linspace(-args.eps_max, args.eps_max, 61)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), sharey=True)
    fig.suptitle("Feeding the trained model mixes shifted along each direction:  "
                 "π(ε) = π₀ + ε·direction", fontweight="bold", fontsize=15)

    for ax, (name, d) in zip(axes, directions):
        max_change = 0.0
        for idx, lab, col in zip(picks, labels, colors):
            orbit = X[idx][None, :] + eps[:, None] * d[None, :]
            pred = predict(orbit)
            max_change = max(max_change, pred.max() - pred.min())
            ax.plot(eps, pred, color=col, lw=2.5, label=lab)
            ax.scatter([0], [y[idx]], color=col, marker="*", s=140,
                       zorder=5, edgecolors="black", linewidths=0.5)
        is_gen = name.startswith("generator")
        ax.set_title(f"{name}\nmax prediction change = {max_change:.2e}"
                     if is_gen else
                     f"{name}\nmax prediction change = {max_change:.2f}",
                     color="#2E7D32" if is_gen else "#B71C1C")
        ax.set_xlabel("shift ε along direction")
        ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("model-predicted σ/σ_ideal")
    axes[0].legend(loc="upper left", title="★ = measured value at ε=0")

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(args.output_dir, "generator_orbits.png")
    os.makedirs(args.output_dir, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {out}")


if __name__ == "__main__":
    main()
