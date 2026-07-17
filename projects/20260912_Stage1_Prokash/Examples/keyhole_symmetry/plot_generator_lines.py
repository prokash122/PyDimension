"""
Simple check that the keyhole scaling-generators work, in the same style
as the concrete example:

  1. take TWO real data points (a low- and a high-e* case);
  2. RESCALE their dimensionless Pi groups along each generator,
     Pi_i -> Pi_i * exp(eps * g_i)   (scaling symmetry = additive in log-Pi);
  3. feed every rescaled point to the trained model;
  4. plot the predicted keyhole eccentricity e*.

Two points x two generators = four lines, and every one is FLAT: the
predicted e* does not move as the Pi groups are rescaled along a
generator. For contrast, each panel also rescales along the Ke direction
(the physics direction the model IS sensitive to); those dashed lines
bend sharply.

This uses the GENUINE trained model saved by discover_symmetry.py
(trained_model.pt: the winning scaling encoder + its jointly trained
decoder) -- no refit. The prediction is
    e* = inverse_minmax( decoder(encoder(Pi)) ),
i.e. the Pi values go straight into the model and e* comes out.

Note on interpretation
----------------------
The scaling encoder computes z = W * log(Pi), and each generator
satisfies W * g = 0, so decoder(encoder(Pi)) cannot see a move along g
-- the flat lines are exact by construction. This confirms the trained
model embodies the discovered scaling symmetry (the rediscovered
keyhole number Ke); it is the model-side consistency check.

Usage
-----
    python plot_generator_lines.py     # after discover_symmetry.py
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

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
from matplotlib.lines import Line2D

# validated palette (dataviz reference instance, light mode)
BLUE = "#2a78d6"
BLUE_DK = "#104281"
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


def describe(g, names):
    order = np.argsort(np.abs(g))[::-1]
    up = [names[j] for j in order if g[j] > 0.15]
    dn = [names[j] for j in order if g[j] < -0.15]
    parts = []
    if up:
        parts.append("Scale up " + ", ".join(up))
    if dn:
        parts.append(("Scale down " if not parts else "scale down ") + ", ".join(dn))
    return "  ·  ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts",
                        default="output_keyhole_symmetry/pipeline_artifacts.npz")
    parser.add_argument("--output-dir", default="output_keyhole_symmetry")
    parser.add_argument("--eps-max", type=float, default=1.0)
    args = parser.parse_args()

    path = args.artifacts
    if not os.path.exists(path):
        path = os.path.join(_here, args.artifacts)
    data = np.load(path, allow_pickle=True)
    pi = data["pi_centred"]                 # (n, 3) centred Pi values
    y = data["y"]                           # measured e*
    gens = np.array(data["generators"])     # (2, 3) log-Pi directions
    ke = np.asarray(data["ke_pi_coords"], dtype=float)   # [0.5, 1, 1]
    names = [str(s) for s in data["pi_names"]]
    y_min = float(np.asarray(data["y_min"]).ravel()[0])
    y_range = float(np.asarray(data["y_range"]).ravel()[0])

    # Load the GENUINE trained model (scaling encoder + its decoder).
    model_path = os.path.join(os.path.dirname(path), "trained_model.pt")
    ckpt = torch.load(model_path, weights_only=False, map_location="cpu")
    encoder = ckpt["encoder"].cpu().eval()
    decoder = ckpt["decoder"].cpu().eval()

    def predict(pi_q):
        """Feed centred Pi values straight through the real model -> e*."""
        with torch.no_grad():
            x = torch.tensor(np.asarray(pi_q), dtype=torch.float32)
            y_norm = decoder(encoder(x)).numpy().ravel()
        return y_norm * y_range + y_min      # invert minmax -> e*

    # sanity: the model reproduces measured e*
    r2 = 1 - np.sum((y - predict(pi)) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"Genuine trained model on data: R2 = {r2:.4f}")

    g_unit = gens / np.linalg.norm(gens, axis=1, keepdims=True)
    ke_unit = ke / np.linalg.norm(ke)        # physics (Ke) direction, output changes

    pred_all = predict(pi)
    order = np.argsort(pred_all)
    idx_lo = order[int(len(order) * 0.15)]
    idx_hi = order[int(len(order) * 0.85)]
    mixes = [("low-e* case", idx_lo, BLUE), ("high-e* case", idx_hi, BLUE_DK)]

    eps = np.linspace(-args.eps_max, args.eps_max, 61)

    # Both generators give exactly flat lines at the same two case levels,
    # so the per-generator panels are visually identical -- draw everything
    # in ONE panel instead.
    fig, ax = plt.subplots(figsize=(10.5, 10.0))
    fig.suptitle("Take a real keyhole case, rescale its Pi groups along a generator,\n"
                 "Ask the model for e*: the predicted eccentricity does not move (flat lines)",
                 fontweight="bold", fontsize=22)

    chg = np.zeros(gens.shape[0])
    for name, idx, col in mixes:
        for gi in range(gens.shape[0]):
            # solid: rescale Pi groups along the generator, Pi -> Pi*exp(eps*g) -> flat
            path_g = pi[idx][None, :] * np.exp(eps[:, None] * g_unit[gi][None, :])
            pred_g = predict(path_g)
            ax.plot(eps, pred_g, lw=3, color=col)
            chg[gi] = max(chg[gi], pred_g.max() - pred_g.min())
        # dashed: rescale along the Ke (physics) direction -> bends
        path_k = pi[idx][None, :] * np.exp(eps[:, None] * ke_unit[None, :])
        ax.plot(eps, predict(path_k), ls="--", lw=2, color=col, alpha=0.85)
        ax.scatter([0], [pred_all[idx]], color=col, marker="*", s=170,
                   zorder=6, edgecolors="white", linewidths=0.8)

    ax.set_title("\n".join(f"g{gi+1}: {describe(g_unit[gi], names)}"
                           for gi in range(gens.shape[0])),
                 fontsize=21)
    print("e* change  " + "   ".join(f"along g{gi+1}: {chg[gi]:.0e}"
                                     for gi in range(gens.shape[0])))
    ax.axvline(0, color=BASE, lw=0.8, ls=":")
    ax.set_xlabel("How far we rescale the Pi groups  (ε)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    handles = [
        Line2D([0], [0], color=INK2, lw=3, ls="-", label="Along a generator (flat)"),
        Line2D([0], [0], color=INK2, lw=2, ls="--", label="Along Ke direction"),
    ]
    ax.set_ylabel("Model-predicted keyhole\neccentricity  e*")
    ax.legend(handles=handles, loc="upper left", frameon=False,
              fontsize=21, handlelength=2.4, borderaxespad=0.3)

    fig.text(0.5, -0.02,
             "★ = A real keyhole case from the dataset (ε=0).\n"
             "Solid = Rescale along a generator (e* held).  "
             "Dashed = Rescale along the Ke direction (e* changes).",
             ha="center", fontsize=20, color="black")

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "generator_lines.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
