"""
One-command runner for the porous-media FULL-Ergun-range experiment.

Runs the whole chain and tees all console output to a single log file you
can share back:

  1. generate_combined_dataset.py  — real LBM (viscous) + synthetic rows
     over the rest of the range (Re_p 1e-3 .. 1e6) from the TEXTBOOK Ergun
     equation (150, 1.75) with noise -> dataset_combined_ergun.csv
  2. plot_full_ergun_range.py       — full curve, LBM points marked
     distinctly (they sit ~0.65x below the textbook viscous branch)
  3. discover_symmetry.py           — Stage1 pipeline on the combined set

Uses the GPU automatically if available (the pipeline device is "auto").
Run it from this directory:

    python run_fullrange_check.py

The combined transcript is written to
    output_porous_fullrange/fullrange_check_full.log
Share that file (and the PNGs) back.
"""

import os
import sys
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_porous_fullrange")
LOG = os.path.join(OUT, "fullrange_check_full.log")

STEPS = [
    [sys.executable, "generate_combined_dataset.py"],
    [sys.executable, "plot_full_ergun_range.py"],
    [sys.executable, "discover_symmetry.py",
     "--data", "dataset_combined_ergun.csv",
     "--seed", "42", "--output-dir", "output_porous_fullrange"],
]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as log:

        def emit(line=""):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            log.write(line + "\n")
            log.flush()

        emit(f"# porous full-range check — {datetime.datetime.now().isoformat()}")
        try:
            import torch
            dev = (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu")
            emit(f"# python {sys.version.split()[0]}, torch {torch.__version__}, "
                 f"cuda={torch.cuda.is_available()} ({dev})")
        except Exception as e:  # pragma: no cover
            emit(f"# torch import failed: {e}")

        for cmd in STEPS:
            emit()
            emit("$ " + " ".join(os.path.basename(c) if c.endswith(".py") else c
                                 for c in cmd))
            proc = subprocess.Popen(
                cmd, cwd=HERE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            proc.wait()
            if proc.returncode != 0:
                emit(f"[step FAILED with exit code {proc.returncode} — stopping]")
                break
        else:
            emit()
            emit("All steps completed.")

        emit(f"\nFull transcript: {LOG}")
        emit("Figures/outputs in: " + OUT + " and this directory")


if __name__ == "__main__":
    main()
