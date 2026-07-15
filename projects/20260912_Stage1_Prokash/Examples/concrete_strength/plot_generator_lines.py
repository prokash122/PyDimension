"""
Simplest possible check that the generators work:

  1. take TWO real mixes from the dataset (a weaker and a stronger one);
  2. change their recipe step by step ALONG each generator;
  3. feed every synthetic recipe to the trained model;
  4. plot the predicted strength.

Two mixes x three generators = six lines, and every one is FLAT: the
model's predicted strength does not move as we walk along a generator.
For contrast, each panel also walks the same mix along the model's
"strength direction" (dashed): that line bends sharply.

Note on interpretation
----------------------
The model predicts the strength residual y = sigma / sigma_ideal, and it
computes y = f(W*pi). Each generator satisfies W*g = 0, so f cannot see a
move along g -- the flat lines are therefore exact by construction. This
figure confirms the trained model genuinely embodies the discovered
symmetry; the independent proof that REAL concrete shares it is the
measured mix-pair test (validate_generators.py / publication figure).

Usage
-----
    python plot_generator_lines.py     # after discover_symmetry_dimensionless.py
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

# validated palette (dataviz reference instance, light mode)
BLUE = "#2a78d6"
BLUE_DK = "#104281"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "text.color":      INK,
    "axes.labelcolor": INK2,
    "xtick.color":     INK2,
    "ytick.color":     INK2,
    "axes.edgecolor":  BASE,
})

_here = os.path.dirname(os.path.abspath(__file__))

PI_LABELS = ["cement/binder-side w/b", "fly ash", "slag", "superplasticizer",
             "coarse agg.", "fine agg.", "age"]
SHORT = ["w/b", "fly ash", "slag", "SP", "coarse agg", "fine agg", "age"]


def fit_head(z, y, seed=0):
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
        loss_fn(model(zt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(zt).numpy().ravel()
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"Decoder head refit on frozen z: R2 = {r2:.4f}")
    return model


def describe(g):
    """Plain-words summary of a generator's biggest moves."""
    order = np.argsort(np.abs(g))[::-1]
    up = [SHORT[j] for j in order[:3] if g[j] > 0.15]
    dn = [SHORT[j] for j in order[:3] if g[j] < -0.15]
    parts = []
    if up:
        parts.append("more " + ", ".join(up))
    if dn:
        parts.append("less " + ", ".join(dn))
    return "  ·  ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts",
                        default="output_concrete_dimensionless/pipeline_artifacts.npz")
    parser.add_argument("--output-dir", default="output_concrete_dimensionless")
    parser.add_argument("--eps-max", type=float, default=1.0)
    args = parser.parse_args()

    path = args.artifacts
    if not os.path.exists(path):
        path = os.path.join(_here, args.artifacts)
    data = np.load(path, allow_pickle=True)
    X = data["X_norm"]
    y = data["y"]
    W = data["W"]
    gens = data["generators"]

    head = fit_head(X @ W.T, y)

    def predict(Xq):
        with torch.no_grad():
            return head(torch.tensor(Xq @ W.T, dtype=torch.float32)).numpy().ravel()

    g_unit = gens / np.linalg.norm(gens, axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(W)
    v1 = Vt[0]

    # two real mixes: a weaker and a stronger one, chosen so the model's
    # prediction (the quantity plotted) is well separated, so each star
    # sits exactly on its own flat line.
    pred_all = predict(X)
    order = np.argsort(pred_all)
    idx_lo = order[int(len(order) * 0.15)]
    idx_hi = order[int(len(order) * 0.85)]
    mixes = [("weaker real mix", idx_lo, BLUE),
             ("stronger real mix", idx_hi, BLUE_DK)]

    eps = np.linspace(-args.eps_max, args.eps_max, 61)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    fig.suptitle("Take a real mix, change its recipe along a generator, ask the model its strength:\n"
                 "the predicted strength does not move (flat lines)",
                 fontweight="bold", fontsize=13.5)

    for gi, ax in enumerate(axes):
        g = g_unit[gi]
        # contrast: strength direction, from the weaker mix
        pred_str = predict(X[idx_lo][None, :] + eps[:, None] * v1[None, :])
        ax.plot(eps, pred_str, ls="--", lw=1.8, color=RED, alpha=0.9,
                label="along strength direction" if gi == 0 else None)

        for name, idx, col in mixes:
            pred = predict(X[idx][None, :] + eps[:, None] * g[None, :])
            ax.plot(eps, pred, lw=3, color=col,
                    label=name if gi == 0 else None)
            ax.scatter([0], [pred_all[idx]], color=col, marker="*", s=170,
                       zorder=6, edgecolors="white", linewidths=0.8)

        rng = predict(X[idx_lo][None, :] + eps[:, None] * g[None, :])
        chg = rng.max() - rng.min()
        ax.set_title(f"generator g{gi+1}\n{describe(g)}", fontsize=10.5)
        ax.text(0.5, 0.06, f"strength change along g{gi+1}: {chg:.0e}",
                transform=ax.transAxes, ha="center", color=INK2, fontsize=9.5)
        ax.axvline(0, color=BASE, lw=0.8, ls=":")
        ax.set_xlabel("how far we change the recipe  (ε)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("model-predicted strength\n(σc ÷ baseline)")
    axes[0].legend(loc="center left", frameon=False, fontsize=9.5)
    axes[0].set_ylim(0.3, 1.6)

    fig.text(0.5, -0.02,
             "★ = a real mix from the dataset (ε=0).  Solid = walk along a generator (strength held).  "
             "Dashed = walk along the strength direction (strength changes).",
             ha="center", fontsize=9.5, color=INK2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "generator_lines.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
