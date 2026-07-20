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
The model predicts the dimensionless strength sigma_c* = sigma/sigma_ideal,
and it computes sigma_c* = f(W*pi). Each generator satisfies W*g = 0, so f
cannot see a move along g -- the flat lines are therefore exact by
construction. This figure confirms the trained model genuinely embodies
the discovered symmetry directions.

Usage
-----
    python plot_generator_lines.py     # after discover_symmetry_dimensionless.py
"""

import os
import argparse

import sys
import numpy as np
import torch

# Make the Stage1 package importable so torch.load can unpickle SymmetryEncoder.
_here0 = os.path.dirname(os.path.abspath(__file__))
for _c in [os.path.join(_here0, "..", ".."),
           os.path.join(_here0, "..", "..", "projects", "20260912_Stage1_Prokash")]:
    _c = os.path.abspath(_c)
    if os.path.isdir(os.path.join(_c, "symmetry_discovery")):
        sys.path.insert(0, _c)
        break

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
    "font.size":       23,
    "axes.titlesize":  25,
    "axes.labelsize":  25,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 21,
    "text.color":      "black",
    "axes.labelcolor": "black",
    "xtick.color":     "black",
    "ytick.color":     "black",
    "axes.edgecolor":  BASE,
})

_here = os.path.dirname(os.path.abspath(__file__))

PI_LABELS = ["w/b", "cement", "slag", "fly ash", "superplasticizer",
             "coarse agg.", "fine agg.", "age"]
SHORT = ["w/b", "cement", "slag", "fly ash", "SP", "coarse agg", "fine agg",
         "age"]


def describe(g):
    """Plain-words summary of a generator's biggest moves."""
    order = np.argsort(np.abs(g))[::-1]
    up = [SHORT[j] for j in order[:3] if g[j] > 0.15]
    dn = [SHORT[j] for j in order[:3] if g[j] < -0.15]
    parts = []
    if up:
        parts.append("More " + ", ".join(up))
    if dn:
        parts.append(("Less " if not parts else "less ") + ", ".join(dn))
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
    y_mean = float(np.asarray(data["y_mean"]).ravel()[0])
    y_std = float(np.asarray(data["y_std"]).ravel()[0])

    # Load the GENUINE trained model (translational encoder + its decoder).
    model_path = os.path.join(os.path.dirname(path), "trained_model.pt")
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    encoder = ckpt["encoder"].cpu().eval()
    decoder = ckpt["decoder"].cpu().eval()

    def predict(Xq):
        """Feed standardized Pi features straight through the real model -> sigma/sigma_ideal."""
        with torch.no_grad():
            x = torch.tensor(np.asarray(Xq), dtype=torch.float32)
            y_norm = decoder(encoder(x)).numpy().ravel()
        return y_norm * y_std + y_mean      # invert standard-scaler -> residual

    r2 = 1 - np.sum((y - predict(X)) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"Genuine trained model on data: R2 = {r2:.4f}")

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

    # All per-generator panels are visually identical (the solid generator
    # walks are flat at the same two mix levels and the dashed strength
    # walk is the same v1 in each), so draw everything in ONE panel.
    fig, ax = plt.subplots(figsize=(10.5, 10.0))
    fig.suptitle("Take a real mix, change its recipe along a generator,\n"
                 "Ask the model its strength: the predicted strength does not move (flat lines)",
                 fontweight="bold", fontsize=22)

    chg = np.zeros(gens.shape[0])
    for name, idx, col in mixes:
        for gi in range(gens.shape[0]):
            # solid: walk along the generator -> flat
            pred_g = predict(X[idx][None, :] + eps[:, None] * g_unit[gi][None, :])
            ax.plot(eps, pred_g, lw=3, color=col)
            chg[gi] = max(chg[gi], pred_g.max() - pred_g.min())
        # dashed: walk along the strength direction -> bends
        pred_s = predict(X[idx][None, :] + eps[:, None] * v1[None, :])
        ax.plot(eps, pred_s, ls="--", lw=2, color=col, alpha=0.85)
        ax.scatter([0], [pred_all[idx]], color=col, marker="*", s=170,
                   zorder=6, edgecolors="white", linewidths=0.8)

    print("strength change  " + "   ".join(f"along g{gi+1}: {chg[gi]:.0e}"
                                           for gi in range(gens.shape[0])))
    ax.set_title("\n".join(f"g{gi+1}: {describe(g_unit[gi])}"
                           for gi in range(gens.shape[0])),
                 fontsize=21)
    ax.axvline(0, color=BASE, lw=0.8, ls=":")
    ax.set_xlabel("How far we change the recipe  (ε)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=INK2, lw=3, ls="-", label="Along a generator (flat)"),
        Line2D([0], [0], color=INK2, lw=2, ls="--", label="Along strength direction"),
    ]
    ax.set_ylabel("Model-predicted σc*")
    ax.set_ylim(0.3, 1.95)
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=21, ncol=1, handlelength=2.4, borderaxespad=0.3)

    fig.text(0.5, -0.02,
             "★ = A real mix from the dataset (ε=0).\n"
             "Solid = Walk along a generator (strength held).  "
             "Dashed = Walk along the strength direction (strength changes).",
             ha="center", fontsize=20, color="black")

    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "generator_lines.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
