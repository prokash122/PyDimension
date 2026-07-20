"""
Standalone publication figure: the symmetry-type competition.

Three encoder families (translational, rotational, scaling) are trained
against the same decoder; the one with the lowest held-out MSE wins. This
script reads the recorded losses from pipeline_artifacts.npz and draws the
bar chart on its own (green = winner, orange = the rest).

Usage
-----
    python plot_symmetry_type.py     # after discover_symmetry_dimensionless.py
"""

import os
import argparse

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
except (AttributeError, ImportError):
    pass
import matplotlib.pyplot as plt

# validated palette (dataviz reference instance, light mode)
GREEN = "#55A868"     # winner
ORANGE = "#DD8452"    # runners-up

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       15,
    "axes.titlesize":  17,
    "axes.labelsize":  16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
})

_here = os.path.dirname(os.path.abspath(__file__))

# Fixed display order (winner first, then the two runners-up).
ORDER = ["translational", "rotational", "scaling"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts",
                        default="output_concrete_dimensionless/pipeline_artifacts.npz")
    parser.add_argument("--output-dir", default="output_concrete_dimensionless")
    args = parser.parse_args()

    path = args.artifacts
    if not os.path.exists(path):
        path = os.path.join(_here, args.artifacts)
    data = np.load(path, allow_pickle=True)

    if "sym_losses" not in data:
        raise SystemExit(
            "pipeline_artifacts.npz has no symmetry losses — re-run "
            "discover_symmetry_dimensionless.py to regenerate it.")

    losses = {str(t): float(v)
              for t, v in zip(data["sym_types"], data["sym_losses"])}
    winner = str(data["winner_type"]) if "winner_type" in data \
        else min(losses, key=losses.get)

    types = [t for t in ORDER if t in losses] + \
            [t for t in losses if t not in ORDER]
    vals = [losses[t] for t in types]
    colors = [GREEN if t == winner else ORANGE for t in types]

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    bars = ax.bar(types, vals, color=colors, edgecolor="black", lw=1)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Validation MSE")
    ax.set_title(f"Symmetry Type (winner: {winner})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{v:.4f}", ha="center", va="bottom", fontsize=13)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "symmetry_type.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"winner: {winner}; losses: "
          + ", ".join(f"{t}={losses[t]:.4f}" for t in types))
    print(f"Saved {out} (+ .pdf)")


if __name__ == "__main__":
    main()
