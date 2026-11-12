# Contributing

Thanks for helping make reward debugging less mysterious. Good contributions include a small reproducible reward failure, reviewed contract, explicit importer, actual reference fixture or clearer documentation.

## Setup

Use Python 3.11/3.12 and Node.js 22+. macOS/Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
npm --prefix web ci
npx --prefix web playwright install chromium
./scripts/check.sh
```

On Windows PowerShell, avoid environment activation:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m ruff check ptm scripts tests examples
.\.venv\Scripts\python.exe -m mypy ptm
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts/smoke_install.py
npm --prefix web ci
npm --prefix web run lint
npm --prefix web run build
```

The complete browser test server script uses POSIX `.venv/bin/python`; run that suite in WSL2 on Windows. The separate install-smoke workflow checks the native Windows installed application. macOS can use `PTM_BROWSER_CHANNEL=chrome ./scripts/check.sh` with installed Chrome.

Frontend assets in `ptm/web` ship in the wheel and source download. Rebuild them after UI changes. Ordinary users need no Node.js. `scripts/build_release.py` builds a clean wheel and verifies packaged bytes; it requires `uv` and assumes the frontend has already been built.

## Evidence rules

Schema changes need finite-value, foreign-key, legacy-hash and partial-group tests. Numerical changes need independently sourced reference output and exact framework provenance. Never generate expected goldens using PTM's own kernel. New adapters need explicit capability/version declarations, an actual fixture and a fail-closed unsupported-version path.

Use the optional `.[trl]` environment only when testing the pinned native reference. See docs/SECURITY.md for its dependency advisories. Native Windows deep capture and GPU training are not part of the verified scope.

## Issues and data

Please provide versions, recipe, error code, reproduction command and a minimal shareable trace. Keep private workspaces, keys, original prompts and checkpoints out of public issues. Synthetic fixtures must be labeled; fixed-answer reward changes must not be presented as future model improvements.

Mechanism figures have editable `.mmd` + `.md` sources in `figures/`. To regenerate, install Mermaid CLI and run `mmdc -i figures/ptm-workflow.mmd -o figures/ptm-workflow.png -b white`. Review arrow directions, clipping and labels before committing.
