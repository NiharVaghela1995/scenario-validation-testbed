"""Reproduction test: recompute everything from scratch and compare with the published numbers.

The expected values in tests/data/published_numbers.json are stored separately from the code that produces the results;
this test never reads the committed results/ folder. Replay numbers are skipped when esmini is not available.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import check_numbers as CN  # noqa: E402

NUMBERS = CN.load_published()


@pytest.fixture(scope="session")
def results(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("repro")
    env = dict(os.environ, TESTBED_RESULTS=str(tmp), TESTBED_FIGURES=str(tmp / "figures"), TESTBED_DATA=str(tmp / "data"),
               TESTBED_SCREENING_OUT=str(tmp / "screening"), MPLBACKEND="Agg")
    for script in ("critical_events/run_critical_events.py", "experiments/fusion_gate.py", "experiments/generation_cutin.py",
                   "experiments/power_heldout.py", "experiments/make_figures.py"):
        args = [sys.executable, script] + (["--mock"] if "run_critical" in script else [])
        subprocess.run(args, cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
    return tmp


@pytest.mark.parametrize("entry", NUMBERS, ids=[e["label"] for e in NUMBERS])
def test_published_number(results, entry):
    if entry.get("needs_esmini") and not CN.replay_ran(results):
        pytest.skip("esmini not found (ESMINI_BIN)")
    got = CN.lookup(results, entry)
    assert CN.within(entry, got), f"{entry['label']}: published {entry['value']}, computed {got}"


def test_every_figure_is_produced(results):
    names = {p.name for p in (results / "figures").glob("*.png")}
    assert {"fig1_cutin_events.png", "fig2_likelihood_vs_tail.png", "fig3_lambda_corner.png", "fig4_power.png",
            "fig5_importance_sampling.png"} <= names
