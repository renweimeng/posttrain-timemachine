# Architecture

The data plane is separate from the UI. Explicit-ID capture events are written
to per-rank append-only JSONL spools. Import validates Pydantic entities and
foreign keys, deduplicates idempotent event IDs, and writes typed Parquet tables
plus complete JSON payloads. The original canonical payload remains available
for auditing fields that are not analytical columns.

Snapshots are content-addressed, committed by staging-directory rename and
published through an atomically replaced manifest. Readers resolve one committed
snapshot; they never combine files from different revisions. Checksums and the
canonical logical snapshot hash are checked by `doctor`/analysis. A crash may
leave a hidden `.pending-*` directory, which is never treated as committed data.

DuckDB executes parameterized filters against immutable Parquet. Its in-memory
catalog is disposable; Parquet and manifests remain the source of truth.
Generation step, optimizer step and policy version remain distinct. Metric
alignment uses explicitly supplied generation-step metadata only.

The computation layer contains no HTTP dependencies. A closed RewardSpec AST
is interpreted without eval/exec. Recipe, filter, input snapshot and spec define
an analysis identity. Jobs transition queued → running → succeeded/failed/cancelled;
only complete result directories are published. Reports embed hashes and state
the evidence boundary. API writes are restricted to the service's own origin.

React is compiled into local static assets and served by FastAPI. There are no
runtime CDNs or external fonts. Text uses React text rendering. L1 token/loss
requests fail explicitly; L2 Canvas surfaces come from captured tokenizer pieces.
Synthetic stress and Arena fixtures are explicitly labeled.

## Limits

The workspace has a single-host worker, not a distributed service. A service restart marks
unfinished jobs failed and allows explicit retry. Large imports are capped at
128 MiB per file and 2 MiB per JSONL line. Split larger capture spools before
importing. No ZIP extraction or arbitrary SQL/code endpoints are provided.
Parquet input is accepted only through committed, verified PTM snapshots.


Token/Objective/Trajectory Parquet tables are additive in schema 0.2.0. Legacy
0.1.0 snapshots omit empty new arrays from the logical hash, preserving their
original identity. Metadata queries read only metadata tables; token arrays load
on explicit inspection/replay paths. Token tables use smaller row groups.
Analysis requests and states persist before computation. Browser URLs identify
run/group/analysis; versioned localStorage contains drafts, not authoritative
results. Server restart marks interrupted jobs failed and retains retry inputs.

Deep capture and model update execution are opt-in modules; app startup does not
import Torch. Package metadata controls the replay button's reference-environment
availability. Numeric checkpoints live in a separate bounded safetensors capture
store and are never included in an ordinary report. Offline bundles optionally
carry token tensors, objectives and independently recomputable scalar objective
evidence, with an exact checksummed file list. The operator chooses text retention.
