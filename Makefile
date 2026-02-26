.PHONY: help setup venv install dev lint test eval app run-ops run-analytics clean

PYTHON := python
VENV := .venv
PIP := $(VENV)/Scripts/pip.exe
RUN := $(VENV)/Scripts/python.exe

help:
	@echo "Common commands:"
	@echo "  make setup         Create venv + install dev deps"
	@echo "  make lint          Run ruff"
	@echo "  make test          Run pytest"
	@echo "  make eval          Placeholder eval (Day 4)"
	@echo "  make run-ops       Placeholder (Day 5)"
	@echo "  make run-analytics Placeholder (Day 7)"
	@echo "  make app           Run Streamlit minimal app"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install -U pip
	$(PIP) install -e ".[dev]"

setup: venv install
dev: setup

lint:
	$(RUN) -m ruff check .

test:
	$(RUN) -m pytest

eval:
	@echo "Eval harness will be added on Day 4. For now: make test"

run-ops:
	@echo "Ops Copilot CLI will be added on Day 5."

run-analytics:
	@echo "Analytics Agent CLI will be added on Day 7."

app:
	$(RUN) -m streamlit run apps/streamlit_app.py

clean:
	rmdir /s /q $(VENV) 2>NUL || exit 0