#!/usr/bin/env python3
"""One command: regenerate every result and figure, then compare with the published numbers.

    python reproduce.py            # 2 to 3 minutes on a laptop, CPU only
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = Path(os.environ.get("TESTBED_RESULTS", ROOT / "results"))


def run(args, log=None):
    print(">", " ".join(args), flush=True)
    env = dict(os.environ, TESTBED_SCREENING_OUT=str(RES / "screening"), MPLBACKEND="Agg")
    with open(log, "w") if log else open(os.devnull, "w") as out:
        subprocess.run([sys.executable, *args], cwd=ROOT, env=env, check=True, stdout=out if log else None)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    run(["critical_events/run_critical_events.py", "--mock"], RES / "screening_mock.log")
    run(["experiments/fusion_gate.py"])
    run(["experiments/generation_cutin.py"], RES / "generation_cutin.txt")
    run(["experiments/power_heldout.py"], RES / "power_heldout.stdout.log")
    (RES / "power_heldout.stdout.log").unlink()
    run(["experiments/make_figures.py"])
    return subprocess.call([sys.executable, "experiments/check_numbers.py", str(RES)], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
