"""Run notebooks/tail_fidelity.ipynb from a clean kernel and compare its numbers with the published evidence.

Usage: pytest tests/test_tail_notebook.py   (packages: notebooks/requirements.txt, a separate environment; skipped without them)
The neural flow is the only model whose result can shift with the torch build, so it has wider tolerances.
"""
import json
import os
import pathlib
import time

import pytest

pytest.importorskip("pyvinecopulib")
nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXPECTED = json.loads((pathlib.Path(__file__).parent / "data" / "tail_fidelity_expected.json").read_text())
TOL = {"loglik": 0.01, "lambda": 0.003, "critical_share": 0.002}
FLOW_TOL = {"loglik": 0.15, "lambda": 0.04, "critical_share": 0.02}
IS_TOL = {"estimate": 0.0005, "se": 0.0005, "ess": 5, "reference": 0.0005, "crude": 0.002}
MAX_SECONDS = 300


def run(tmp):
    out = tmp / "results.json"
    nb = nbformat.read(ROOT / "notebooks/tail_fidelity.ipynb", as_version=4)
    os.environ["TAIL_RESULTS_JSON"] = str(out)
    t0 = time.time()
    NotebookClient(nb, timeout=MAX_SECONDS, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
    return json.loads(out.read_text()), time.time() - t0


def test_notebook_matches_evidence(tmp_path):
    got, seconds = run(tmp_path)
    assert seconds < MAX_SECONDS
    bad = []
    for model, values in EXPECTED.items():
        tol = IS_TOL if model == "importance_sampling" else FLOW_TOL if model == "Normalizing flow" else TOL
        for key, want in values.items():
            limit = tol.get(key, 0.001)
            if abs(got[model][key] - want) > limit:
                bad.append(f"{model} {key}: got {got[model][key]:.5f}, published {want:.5f}, tolerance {limit}")
    assert not bad, "\n".join(bad)
    rejected = [k for k, v in got.items() if isinstance(v, dict) and "p_lambda" in v and v["p_lambda"] < 0.05]
    assert len(rejected) == 7, rejected
