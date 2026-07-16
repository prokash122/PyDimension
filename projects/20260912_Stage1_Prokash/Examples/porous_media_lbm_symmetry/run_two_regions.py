"""
One-command runner for the two-region porous-media symmetry experiment.

Runs the whole chain, teeing each pipeline's console output to a run.log
inside its own output directory:

  1. generate_inertial_dataset.py — synthetic inertia-dominated points from
     the actual Ergun formula + noise  ->  dataset_ergun_inertial.csv
  2. plot_two_regions.py           — both regions on the Ergun master curve
     ->  ergun_two_regions.png
  3. discover_symmetry.py on the VISCOUS region (real LBM data)
     ->  output_viscous_region/
  4. discover_symmetry.py on the INERTIAL region (synthetic Ergun data)
     ->  output_inertial_region/

Threads and the hash seed are locked for bit-reproducibility
(OMP/MKL/OPENBLAS_NUM_THREADS=1, PYTHONHASHSEED=0), matching the committed
run.log files.

Usage (from this directory):

    python run_two_regions.py                 # both regions
    python run_two_regions.py --noise 0.2     # noisier synthetic data
    python run_two_regions.py --region viscous    # just one region
    python run_two_regions.py --region inertial
"""

import os
import sys
import argparse
import subprocess
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))

PIPE_ARGS = ["--seed", "42", "--latent-epochs", "300",
             "--sym-epochs", "600", "--n-restarts", "3"]

REPRO_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
}


def run_step(cmd, log_path=None):
    print("\n$ " + " ".join(os.path.basename(c) if c.endswith(".py") else c
                            for c in cmd))
    env = dict(os.environ, **REPRO_ENV)
    log = open(log_path, "w", encoding="utf-8") if log_path else None
    proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        if log:
            log.write(line)
            log.flush()
    proc.wait()
    if log:
        log.close()
    if proc.returncode != 0:
        print(f"[step FAILED with exit code {proc.returncode} — stopping]")
        sys.exit(proc.returncode)


def main():
    ap = argparse.ArgumentParser(description="Run the two-region experiment")
    ap.add_argument("--noise", type=float, default=0.05,
                    help="Synthetic inertial-data noise width "
                         "(0.05=~5%%, 0.2=~22%%). Default 0.05")
    ap.add_argument("--region", choices=["both", "viscous", "inertial"],
                    default="both",
                    help="Which region's pipeline to run. Default both")
    args = ap.parse_args()

    print(f"# porous two-region check — {datetime.datetime.now().isoformat()}")

    if args.region in ("both", "inertial"):
        run_step([sys.executable, "generate_inertial_dataset.py",
                  "--noise", str(args.noise)])
        run_step([sys.executable, "plot_two_regions.py"])

    if args.region in ("both", "viscous"):
        out = os.path.join(HERE, "output_viscous_region")
        os.makedirs(out, exist_ok=True)
        run_step([sys.executable, "discover_symmetry.py",
                  "--data", "dataset_lbm_porous.csv",
                  "--output-dir", "output_viscous_region"] + PIPE_ARGS,
                 log_path=os.path.join(out, "run.log"))

    if args.region in ("both", "inertial"):
        out = os.path.join(HERE, "output_inertial_region")
        os.makedirs(out, exist_ok=True)
        run_step([sys.executable, "discover_symmetry.py",
                  "--data", "dataset_ergun_inertial.csv",
                  "--output-dir", "output_inertial_region"] + PIPE_ARGS,
                 log_path=os.path.join(out, "run.log"))

    print("\nAll steps completed.")
    print("Viscous-region outputs:  output_viscous_region/")
    print("Inertial-region outputs: output_inertial_region/")


if __name__ == "__main__":
    main()
