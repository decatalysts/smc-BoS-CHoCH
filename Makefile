.PHONY: setup test-run

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test-run:
	. .venv/bin/activate && python test_smc.py
