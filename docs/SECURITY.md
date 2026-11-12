# Local security boundary and dependency scans

The service binds loopback and rejects unexpected Host and cross-origin mutation
requests. There is no remote sharing mode or arbitrary Python/SQL endpoint.
Reward rules use a closed AST. Import validates finite values, IDs, schema,
foreign keys, file/line sizes, nesting and YAML alias graphs. Snapshot and report
paths are constrained; report ZIPs are verified in memory, never extracted.
Captured output is rendered as text, with a restrictive production CSP.

On 2026-11-03 the production frontend (`npm audit --omit=dev`) and the complete
default Python runtime dependency closure (`pip-audit` against
`requirements-runtime.lock`) reported **zero known advisories**. This is a scan
result, not a security certification. Public scan JSON is in `docs/benchmarks/`.

The optional `.[trl]` reference environment is deliberately pinned to TRL
0.24.0 / Transformers 4.57.1 / Torch 2.9.0 / Accelerate 1.11.0 so that numerical
fixtures have a fixed executable source. Scanning the full test environment
reported **16 advisory entries across 3 optional packages** (Accelerate, Torch,
Transformers; some aliases describe the same issue). Several concern loading
untrusted checkpoints, remote model code or conversion utilities. See
`docs/benchmarks/pip-audit.json` for exact identifiers and descriptions.

These libraries are **not default PTM dependencies**, and neither the CLI demo
nor the server imports them. The supplied reference script runs offline with a
locally constructed random model and tokenizer, no external checkpoint and no
resume/conversion/custom-code path. Do not use that pinned reference extra to
load unknown models or checkpoints. A production training environment needs its
own current dependency review; reference parity does not establish security of
an arbitrary trainer deployment. No Python advisories were suppressed to make
the scan green.

Tests cover browser XSS text rendering, invalid Origin/Host, chunked request
limits, unsafe paths and symlinks, archive traversal, hash tampering, YAML aliases,
untrusted DSL types, missing capture inputs and preservation of trainer rewards
when capture fails. No cloud service, telemetry or outbound runtime request is
part of the product. Development tools and dependency installation have their
own network behavior; Storybook telemetry is disabled in project config.
