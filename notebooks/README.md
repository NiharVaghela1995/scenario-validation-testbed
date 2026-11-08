# Notebooks

`tail_fidelity.ipynb` walks through the cut-in tail comparison on a laptop CPU, with nine models on the G2 generator. All data are synthetic and generated inside the notebook, and every result is Illustrative (synthetic).

Its package pins differ from the main `requirements.txt`, so it has its own environment (tested on Python 3.10):

```bash
python3.10 -m venv .venv-notebook
.venv-notebook/bin/pip install -r notebooks/requirements.txt
.venv-notebook/bin/python -m pytest -q tests/test_tail_notebook.py
```

The test runs the notebook from a clean kernel and compares its numbers with the published ones.
