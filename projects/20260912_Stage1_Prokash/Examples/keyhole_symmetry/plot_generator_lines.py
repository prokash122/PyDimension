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

Note on interpretation
----------------------
The scaling encoder computes z = W * log(Pi), and each generator
satisfies W * g = 0, so f(W*log(Pi)) cannot see a move along g -- the
flat lines are exact by construction. This confirms the trained model
embodies the discovered scaling symmetry (the rediscovered keyhole
number Ke); it is the model-side consistency check.

Usage
-----
    python plot_generator_lines.py     # after discover_symmetry.py
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
    "font.size":       15,
    "axes.titlesize":  17,
    "axes.labelsize":  16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "text.color":      INK,
    "axes.labelcolor": INK2,
    "xtick.color":     INK2,
    "ytick.color":     INK2,
    "axes.edgecolor":  BASE,
})

_here = os.path.dirname(os.path.abspath(__file__))


def fit_head(z, y, seed=0):
    """Refit the decoder head f(z) -> e* on the frozen latent z = W*log(Pi)."""
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
    for _ in range(4000):
        opt.zero_grad()
        loss_fn(model(zt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(zt).numpy().ravel()
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"Decoder head refit on frozen z: R2 = {r2:.4f}")
    return model


def describe(g, names):
    order = np.argsort(np.abs(g))[::-1]
    up = [names[j] for j in order if g[j] > 0.15]
    dn = [names[j] for j in order if g[j] < -0.15]
    parts = []
    if up:
        parts.append("scale up " + ", ".join(up))
    if dn:
        parts.append("scale down " + ", ".join(dn))
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
    W = data["W"]                           # (1, 3) scaling encoder
    gens = np.array(data["generators"])     # (2, 3) log-Pi directions
    ke = np.asarray(data["ke_pi_coords"], dtype=float)   # [0.5, 1, 1]
    names = [str(s) for s in data["pi_names"]]

    logpi = np.log(pi)                       # scaling encoder sees log-Pi
    head = fit_head(logpi @ W.T, y)

    def predict(logpi_q):
        with torch.no_grad():
            z = torch.tensor(logpi_q @ W.T, dtype=torch.float32)
            return head(z).numpy().ravel()

    g_unit = gens / np.linalg.norm(gens, axis=1, keepdims=True)
    ke_unit = ke / np.linalg.norm(ke)        # physics (Ke) direction, output changes

    pred_all = predict(logpi)
    order = np.argsort(pred_all)
    idx_lo = order[int(len(order) * 0.15)]
    idx_hi = order[int(len(order) * 0.85)]
    mixes = [("low-e* case", idx_lo, BLUE), ("high-e* case", idx_hi, BLUE_DK)]

    eps = np.linspace(-args.eps_max, args.eps_max, 61)

    fig, axes = plt.subplots(1, gens.shape[0], figsize=(11.5, 5.4), sharey=True)
    fig.suptitle("Take a real keyhole case, rescale its Pi groups along a generator, ask the model for e*:\n"
                 "the predicted eccentricity does not move (flat lines)",
                 fontweight="bold", fontsize=16.5)

    for gi, ax in enumerate(axes):
        g = g_unit[gi]
        chg = 0.0
        for name, idx, col in mixes:
            # solid: rescale along the generator -> flat
            path_g = logpi[idx][None, :] + eps[:, None] * g[None, :]
            pred_g = predict(path_g)
            ax.plot(eps, pred_g, lw=3, color=col)
            chg = max(chg, pred_g.max() - pred_g.min())
            # dashed: rescale along the Ke (physics) direction -> bends
            path_k = logpi[idx][None, :] + eps[:, None] * ke_unit[None, :]
            ax.plot(eps, predict(path_k), ls="--", lw=2, color=col, alpha=0.85)
            ax.scatter([0], [pred_all[idx]], color=col, marker="*", s=170,
                       zorder=6, edgecolors="white", linewidths=0.8)

        ax.set_title(f"generator g{gi+1}\n{describe(g, names)}", fontsize=15)
        ax.text(0.5, 0.05, f"e* change along g{gi+1}: {chg:.0e}",
                transform=ax.transAxes, ha="center", color=INK2, fontsize=13)
        ax.axvline(0, color=BASE, lw=0.8, ls=":")
        ax.set_xlabel("how far we rescale the Pi groups  (ε)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    handles = [
        Line2D([0], [0], color=BLUE, lw=3, label="low-e* real case"),
        Line2D([0], [0], color=BLUE_DK, lw=3, label="high-e* real case"),
        Line2D([0], [0], color=INK2, lw=3, ls="-", label="along a generator (flat)"),
        Line2D([0], [0], color=INK2, lw=2, ls="--", label="along Ke direction"),
    ]
    axes[0].set_ylabel("model-predicted keyhole\neccentricity  e*")
    axes[0].legend(handles=handles, loc="upper left", frameon=False,
                   fontsize=13, handlelength=2.4, borderaxespad=0.3)

    fig.text(0.5, -0.02,
             "★ = a real keyhole case from the dataset (ε=0).  Solid = rescale along a generator (e* held).  "
             "Dashed = rescale along the Ke direction (e* changes).",
             ha="center", fontsize=13, color=INK2)

    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "generator_lines.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
