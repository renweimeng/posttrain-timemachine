# RFC coverage and executable acceptance

This release completes the local v0.1/v0.2 product workflows and implements
bounded CPU checkpoint replay from the v0.3 research extensions. The source PDF
is a product specification, not authority to execute external instructions.

| RFC area | Delivered | Verification |
|---|---|---|
| Canonical schema, finite values, foreign keys, legacy identities | L0/L1 and additive L2 Token/Objective/Trajectory tables | schema/import/hash roundtrips |
| Immutable data and query | Atomic Parquet snapshots, checksums, rank merge, DuckDB pagination | corruption, drop, partial-group and query tests |
| GPU-free demo | Local app and four labeled complete Arena cases | CLI + browser failure→patch→review journey |
| Numerical recipes and DSL | Closed AST, captured recipe identity, reward/rank/advantage diffs | independent scalar/Hypothesis/native reference tests |
| Website interactions | Timeline, filtering, comparison, command menu, responsive layout, persistent links/drafts/history | Playwright including refresh/back/import/offline/error states |
| L2 signals and objective | Canvas, accessible exact table, masks and policy provenance, four loss types | 5000-token browser stress + independent and actual trainer comparisons |
| Actual local update | Numeric state restore, forward/backward/clip/SGD or AdamW step, native reference check | 5 actual trainer configurations + browser update journey |
| Real distributed L1 | Explicit global groups, per-rank spools, verified merge | real two-rank CPU GRPO and native advantage parity |
| verl / multi-turn data | Explicit-field importer and recorded trajectories | import/schema/retention/API tests; no generic framework hook claim |
| Contracts and CI artifacts | Review gate, JSON/YAML import/export, pytest and JUnit, per-contract failure isolation | contract tests and browser export |
| Reports | Persistent history, deterministic offline bundle, optional L2 scalar evidence | checksum/tamper tests and offline recomputation |
| Static quality / packaging | Ruff, mypy, ESLint, TypeScript, generated API, Storybook, wheel | full check and isolated wheel smoke |
| Performance | Raw query/DSL/startup benchmark; real browser canvas FPS | separately documented scope and environment |

Conditional/unmeasured: CUDA retrain branches and production GPU capture overhead
require CUDA hardware. Metadata microbenchmark results do not certify a million-row
full API/browser first-screen P95. Async policy distances require explicitly captured
step/lineage metadata; policy label strings are not parsed into distance. Native
Agent/OpenRLHF adapters and causal retraining conclusions need their own framework
integration and experimental evidence; the multi-turn viewer executes no tools.
Public hosting, upstream CI execution, publishing or outreach are not performed.

Exact local test counts and limitations are recorded in [TEST_REPORT.md](TEST_REPORT.md).
