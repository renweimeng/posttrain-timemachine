#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p output
.venv/bin/ruff check ptm scripts tests examples
.venv/bin/mypy ptm
.venv/bin/pytest --cov=ptm --cov-report=term-missing --cov-report=json:output/coverage.json --junitxml=output/pytest-junit.xml
.venv/bin/python scripts/generate_schemas.py
npm --prefix web run generate:api
npm --prefix web run lint
npm --prefix web run build
npm --prefix web run test:e2e
