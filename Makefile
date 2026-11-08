PY ?= python3.12
.PHONY: reproduce test
reproduce: .venv/bin/python
	.venv/bin/python reproduce.py
test: .venv/bin/python
	.venv/bin/python -m pytest -q tests/
.venv/bin/python: requirements.txt
	$(PY) -m venv .venv
	.venv/bin/pip install -q -r requirements.txt
	touch .venv/bin/python
