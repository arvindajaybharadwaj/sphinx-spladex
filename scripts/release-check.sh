#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m pytest -m 'not integration'
python -m build
python -m twine check dist/*
