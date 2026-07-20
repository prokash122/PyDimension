"""
One-command runner for the concrete translational-generator check.

Runs the full chain and tees all console output to a single log file you
can share back:

  1. discover_symmetry_dimensionless.py  — trains the pipeline, saves the
     GENUINE model (trained_model.pt) + pipeline_artifacts.npz
  2. plot_generator_lines.py             — flat lines: model output held
     constant along each generator (prints the end-to-end model R2)
  3. plot_symmetry_type.py               — validation-MSE bar chart of the
     three competing symmetry families

Uses the GPU automatically if available (the pipeline's device is
"auto"). Run it from this directory:

    python run_generator_check.py

The combined transcript is written to
    output_concrete_dimensionless/generator_check_full.log
Share that file (and, if you like, the PNGs in the same folder).
"""

import os
import sys
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_concrete_dimensionless")
LOG = os.path.join(OUT, "generator_check_full.log")

STEPS = [
    [sys.executable, "discover_symmetry_dimensionless.py",
     "--data", "Concrete_Data.xls", "--seed", "42", "--max-latent", "6"],
    [sys.executable, "plot_generator_lines.py"],
    [sys.executable, "plot_symmetry_type.py"],
]


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as log:

        def emit(line=""):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            log.write(line + "\n")
            log.flush()

        emit(f"# concrete generator check — {datetime.datetime.now().isoformat()}")
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
        emit("Figures in: " + OUT)


if __name__ == "__main__":
    main()
